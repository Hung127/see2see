"""Command-line interface (DESIGN §8/§9).

Subcommands:

* ``run``        — execute a configuration headless, optionally into an
  experiment database and optionally resuming from a snapshot;
* ``reproduce``  — re-execute a stored run and compare the final checksum;
* ``report``     — CSV + plots for a stored run;
* ``viz``        — the pygame application (imported lazily so the CLI works
  without pygame installed);
* ``benchmark``  — measure headless throughput at a given population;
* ``experiments``— run the DESIGN §13 scenarios (scaled down by default).

Exit codes: 0 success, 2 configuration error, 3 snapshot version error.
Unexpected runtime errors are not swallowed: they print a traceback and exit 1.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace

import yaml

from .config import Config, load_config
from .engine import Engine
from .persistence import (
    ExperimentDB, SnapshotVersionError, load_snapshot, save_snapshot,
)
from .stats import (
    StatsRecorder, export_csv, export_lineage_csv, make_plots,
)


class ConfigError(Exception):
    """A configuration file could not be loaded or is invalid."""


def _load_config(path: str) -> Config:
    try:
        return load_config(path)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot load configuration {path!r}: {exc}") from exc


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evolve", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a simulation headless")
    run.add_argument("--config", required=True)
    run.add_argument("--ticks", type=int, default=None,
                     help="total ticks to reach (overrides config.ticks)")
    run.add_argument("--headless", action="store_true",
                     help="accepted for interface compatibility; a run never "
                          "opens a window")
    run.add_argument("--db", default=None, help="experiment database to write")
    run.add_argument("--seed", type=int, default=None, help="override config seed")
    run.add_argument("--resume", default=None,
                     help="snapshot file to resume from (config must match)")
    run.set_defaults(func=_cmd_run)

    rep = sub.add_parser("reproduce", help="re-execute a stored run")
    rep.add_argument("--db", required=True)
    rep.add_argument("--run-id", type=int, required=True)
    rep.set_defaults(func=_cmd_reproduce)

    report = sub.add_parser("report", help="CSV and plots for a stored run")
    report.add_argument("--db", required=True)
    report.add_argument("--run-id", type=int, default=None,
                        help="default: the latest run")
    report.add_argument("--out", required=True)
    report.set_defaults(func=_cmd_report)

    viz = sub.add_parser("viz", help="interactive pygame application")
    viz.add_argument("--config", required=True)
    viz.set_defaults(func=_cmd_viz)

    bench = sub.add_parser("benchmark", help="measure headless throughput")
    bench.add_argument("--population", type=int, required=True)
    bench.add_argument("--ticks", type=int, required=True)
    bench.add_argument("--profile", action="store_true",
                       help="print a cProfile top-10 table for the same point")
    bench.set_defaults(func=_cmd_benchmark)

    exp = sub.add_parser("experiments", help="run DESIGN §13 experiments")
    exp.add_argument("--id", default="all",
                     choices=["all", "a", "b", "c", "d", "e", "f", "g", "h"])
    exp.add_argument("--all", action="store_true",
                     help="run every experiment (same as --id all)")
    exp.add_argument("--out", required=True, help="output directory")
    exp.add_argument("--scale", type=float, default=None,
                     help=f"area/time multiplier (default "
                          f"{0.5}; --full is equivalent to 1.0)")
    exp.add_argument("--full", action="store_true",
                     help="run at design scale (area and horizon x1; ~7.5 h)")
    exp.add_argument("--ticks", type=int, default=None,
                     help="override the scaled horizon")
    exp.add_argument("--seed", type=int, default=None,
                     help="override the base seed (assay replicates use +i)")
    exp.add_argument("--replicates", type=int, default=6,
                     help="assay evolution replicates")
    exp.add_argument("--samples", type=int, default=10,
                     help="genomes sampled per assay replicate")
    exp.add_argument("--arena-ticks", type=int, default=None,
                     help="assay arena horizon (default: 500 x scale)")
    exp.set_defaults(func=_cmd_experiments)

    ctl = sub.add_parser(
        "controllers", help="compare the controller baselines (opt-in evaluation)")
    ctl.add_argument("--out", required=True, help="output directory")
    ctl.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13],
                     help="paired seeds (one run per controller per seed)")
    ctl.add_argument("--ticks", type=int, default=None,
                     help="override the config horizon")
    ctl.add_argument("--population", type=int, default=None,
                     help="override the founder population")
    ctl.add_argument("--config", default=None,
                     help="base configuration (default: configs/default.yaml)")
    ctl.add_argument("--only", nargs="+", default=None,
                     help="controllers to run (default: all)")
    ctl.set_defaults(func=_cmd_controllers)

    sens = sub.add_parser(
        "sensitivity", help="one-factor-at-a-time parameter sensitivity sweeps")
    sens.add_argument("--out", required=True, help="output directory")
    sens.add_argument("--sweep", default="all",
                      help=f"which sweep, or 'all' (default). Known: "
                           f"bite, eta_p, soil_diffusion, sigma_trait, "
                           f"maturity_age, decay")
    sens.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13],
                      help="paired seeds per parameter value")
    sens.add_argument("--ticks", type=int, default=250,
                      help="horizon after scaling (default 250)")
    sens.add_argument("--scale", type=float, default=None,
                      help="area/time multiplier (default: the experiment "
                           "default)")
    sens.add_argument("--config", default=None,
                      help="base configuration (default: configs/default.yaml)")
    sens.set_defaults(func=_cmd_sensitivity)
    return p


def _cmd_run(args) -> int:
    cfg = _load_config(args.config)
    if args.seed is not None:
        cfg = replace(cfg, seed=int(args.seed))
    if args.ticks is not None:
        cfg = replace(cfg, ticks=int(args.ticks))
    target = int(cfg.ticks)

    db = ExperimentDB(args.db) if args.db else None
    run_id = None
    if db is not None:
        exp_id = db.create_experiment("cli-run", cfg, seed=cfg.seed)
        run_id = db.start_run(exp_id)

    try:
        if args.resume:
            try:
                eng = load_snapshot(args.resume, cfg)
            except SnapshotVersionError:
                raise
            except ValueError as exc:
                raise ConfigError(str(exc)) from exc
        else:
            eng = Engine(cfg)

        recorder = StatsRecorder()
        # Ancestry is only collected when it will be stored: it is opt-in so a
        # plain headless run pays nothing for it.
        if db is not None:
            from .stats import LineageRecorder

            eng.lineage = LineageRecorder()
        started = time.perf_counter()
        while eng.tick < target:
            eng.step()
            if eng.tick % max(1, int(cfg.stats.interval)) == 0:
                row = recorder.record(eng)
                if db is not None:
                    db.flush_stats(run_id, [row])
            if (db is not None and cfg.stats.snapshot_interval > 0
                    and eng.tick % int(cfg.stats.snapshot_interval) == 0):
                directory = os.path.dirname(os.path.abspath(args.db))
                snap_path = os.path.join(
                    directory, f"snap_run{run_id}_{eng.tick}.npz")
                manifest = save_snapshot(eng, snap_path)
                db.add_snapshot(run_id, eng.tick, snap_path, manifest["sha256"],
                                manifest["format_version"],
                                rng_state=None, manifest=manifest)
        elapsed = time.perf_counter() - started
        checksum = eng.checksum()
        if db is not None:
            for ev in eng.events.log:
                db.record_event(run_id, ev.start_tick, ev.kind,
                                {"duration": ev.duration,
                                 "magnitude": ev.magnitude,
                                 "region": list(ev.region)})
            # Finished lineage rows (bounded deque) plus still-living organisms,
            # which have no death_tick yet and are written as such rather than
            # given a fabricated one.
            if eng.lineage is not None:
                records = eng.lineage.rows()
                if records:
                    db.append_lineage(run_id, records)
            db.finish_run(run_id, eng.tick, "completed", checksum=checksum)
    finally:
        if db is not None:
            db.close()

    rate = eng.tick / elapsed if elapsed > 0 else 0.0
    print(f"run {run_id if run_id is not None else '-'}: {eng.tick} ticks in "
          f"{elapsed:.1f}s ({rate:.1f} ticks/s), population "
          f"{eng.population.n_alive}, checksum {checksum[:12]}")
    return 0


def _cmd_reproduce(args) -> int:
    db = ExperimentDB(args.db)
    try:
        record = db.run_record(args.run_id)
    finally:
        db.close()
    cfg = record["config"]
    eng = Engine(cfg)
    started = time.perf_counter()
    eng.run(int(cfg.ticks))
    got = eng.checksum()
    stored = record["end_checksum"]
    elapsed = time.perf_counter() - started
    if stored and got != stored:
        print(f"reproduce MISMATCH for run {args.run_id}: stored {stored}, "
              f"re-executed {got}", file=sys.stderr)
        return 1
    print(f"reproduce run {args.run_id}: checksum {got[:12]} matches "
          f"({int(cfg.ticks)} ticks in {elapsed:.1f}s)")
    return 0


def _cmd_report(args) -> int:
    db = ExperimentDB(args.db)
    try:
        if args.run_id is None:
            row = db.conn.execute("SELECT MAX(id) FROM runs").fetchone()
            if row is None or row[0] is None:
                raise ConfigError(f"database {args.db!r} has no runs")
            run_id = int(row[0])
        else:
            run_id = int(args.run_id)
        rows = db.read_stats(run_id)
        lineage = db.read_lineage(run_id)
    finally:
        db.close()
    if not rows:
        raise ConfigError(f"run {run_id} has no stored statistics")
    os.makedirs(args.out, exist_ok=True)
    export_csv(rows, os.path.join(args.out, "stats.csv"))
    if lineage:
        export_lineage_csv(lineage, os.path.join(args.out, "lineage.csv"))
    make_plots(rows, args.out)
    extra = f", {len(lineage)} lineage rows" if lineage else ""
    print(f"report for run {run_id}: {len(rows)} rows{extra} -> {args.out}")
    return 0


def _cmd_viz(args) -> int:
    from .viz.app import launch  # lazy: pygame is an optional dependency
    launch(_load_config(args.config))
    return 0


def _cmd_benchmark(args) -> int:
    from .bench import profile_point, run_point

    result = run_point(args.population, args.ticks)
    print(f"{result['population']} organisms, {result['plants']} plants, "
          f"{result['ticks']} ticks: {result['ticks_per_s']} ticks/s "
          f"(mean population {result['mean_population']})")
    if args.profile:
        print(profile_point(args.population, args.ticks))
    return 0


def _cmd_experiments(args) -> int:
    from .experiments import DEFAULT_SCALE, run_all, run_assay, run_scenario

    scale = 1.0 if args.full else (args.scale if args.scale is not None
                                   else DEFAULT_SCALE)
    experiment_id = "all" if args.all else args.id
    if experiment_id == "all":
        results = run_all(args.out, scale=scale, ticks=args.ticks,
                          assay_replicates=args.replicates,
                          assay_samples=args.samples,
                          assay_arena_ticks=args.arena_ticks)
        g = results["g"]["pooled"]
        print(f"experiments complete: scale {scale}, assay pooled diff "
              f"{g['mean']:+.4f} bu (p = {g['p']:.4f}, n = {g['n']}) "
              f"-> {args.out}")
        return 0
    if experiment_id == "g":
        result = run_assay(args.out, scale=scale, replicates=args.replicates,
                           samples=args.samples, ticks=args.ticks,
                           arena_ticks=args.arena_ticks)
        pooled = result["pooled"]
        print(f"assay: {pooled['n']} paired arenas, mean diff "
              f"{pooled['mean']:+.4f} bu, Wilcoxon p = {pooled['p']:.4f} "
              f"-> {args.out}")
        return 0
    result = run_scenario(experiment_id, args.out, scale=scale,
                          ticks=args.ticks, seed=args.seed)
    print(f"experiment {experiment_id}: {result['ticks']} ticks, final "
          f"population {result['final_population']}, "
          f"extinct = {result['extinct']} -> {args.out}")
    return 0


def _cmd_controllers(args) -> int:
    from .controller_eval import run_controller_comparison

    summary = run_controller_comparison(
        args.out, controllers=args.only, seeds=args.seeds,
        ticks=args.ticks, config_path=args.config,
        population=args.population)
    print(f"controller comparison: {len(summary['rows'])} runs, "
          f"{summary['ticks']} ticks, seeds {summary['seeds']} -> {args.out}")
    for r in summary["rows"]:
        print(f"  {r['controller']:8s} seed {r['seed']:>3d}: final population "
              f"{r['final_population']:4d}, births {r['births']:4d}, "
              f"deaths {r['deaths']:4d}"
              f"{' [EXTINCT]' if r['extinct'] else ''}")
    return 0


def _cmd_sensitivity(args) -> int:
    from .experiments import DEFAULT_SCALE
    from .sensitivity import run_sweep

    scale = args.scale if args.scale is not None else DEFAULT_SCALE
    report = run_sweep(args.out, sweep=args.sweep, seeds=args.seeds,
                       ticks=args.ticks, scale=scale, config_path=args.config)
    print(f"sensitivity: {len(report['rows'])} runs, {report['ticks']} ticks, "
          f"scale {scale}, seeds {report['seeds']} -> {args.out}")
    for key, block in report["sweeps"].items():
        print(f"  {key} ({block['parameter']}), baseline "
              f"{block['baseline']}:")
        for row in block["table"]:
            print(f"    {row['value']:>8}: final population "
                  f"{row['final_population_mean']:.1f} ± "
                  f"{row['final_population_sd']:.1f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.func(args)
    except SnapshotVersionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
