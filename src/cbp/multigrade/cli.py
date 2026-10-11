"""Repeatable offline ABCD simulation; no paid API calls."""
import argparse
import gzip
import json
from pathlib import Path
from .protocol import METHODS
from .simulation import ATTACKS, read_calibration, make_world, run_world


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration",type=Path,default=Path("configs/multigrade-calibration.json"))
    parser.add_argument("--profile-panel",choices=("profiles","boundary_profiles"),default="profiles")
    parser.add_argument("--seeds",type=int,nargs="+",default=[7])
    parser.add_argument("--m",type=int,default=20)
    parser.add_argument("--actual-f",type=int,nargs="+",default=[6])
    parser.add_argument("--F",type=int,default=9,help="Predeclared conservative budget, not inferred from actual f")
    parser.add_argument("--pool-size",type=int,default=512)
    parser.add_argument("--heldout-size",type=int,default=2048)
    parser.add_argument("--looks",type=int,nargs="+",default=[16,32,64,128,256,512])
    parser.add_argument("--rho",type=float,default=.50)
    parser.add_argument("--delta",type=float,default=.05)
    parser.add_argument("--methods",choices=METHODS,nargs="+",default=list(METHODS))
    parser.add_argument("--attacks",choices=ATTACKS,nargs="+",default=["selection_badmouth"])
    parser.add_argument("--gzip",action="store_true",help="Lossless deterministic compressed raw JSON")
    parser.add_argument("--output",type=Path,default=Path("runs/multigrade"))
    args = parser.parse_args(argv)
    calibration,digest = read_calibration(args.calibration)
    if args.profile_panel != "profiles":
        calibration = dict(calibration, profiles=calibration[args.profile_panel])
    args.output.mkdir(parents=True,exist_ok=True)
    summaries = []
    for seed in args.seeds:
        for f in args.actual_f:
            world = make_world(calibration=calibration,m=args.m,N=args.pool_size,heldout_size=args.heldout_size,seed=seed,f=f)
            for attack in args.attacks:
                for method in args.methods:
                    result = run_world(world,F=args.F,rho=args.rho,delta=args.delta,looks=tuple(args.looks),method=method,attack=attack)
                    result["calibration"] = {"sha256":digest,"document":calibration}
                    name = f"seed-{seed}-f-{f}-{attack}-{method}.json"
                    payload = (json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode("utf-8")
                    if args.gzip:
                        name += ".gz"
                        with (args.output/name).open("wb") as file:
                            with gzip.GzipFile(filename="",mode="wb",fileobj=file,compresslevel=6,mtime=0) as compressed:
                                compressed.write(payload)
                    else:
                        (args.output/name).write_bytes(payload)
                    summaries.append({"file":name,"config":result["config"],"metrics":result["metrics"]})
    (args.output/"summary.json").write_text(json.dumps(summaries,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(args.output),"cases":len(summaries)}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
