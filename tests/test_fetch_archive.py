"""pc/fetch.py's "archive" source (tools/plan.py writes one for suites/audio/kws120.json): files taken out of a pinned
.tar.gz by path, a member name's leading "./" not part of the path, each file checked by sha256.

    python3 tests/test_fetch_archive.py      (runs curl on a file:// URL; no network)

Cases: a fresh folder (the files come out, the archive is deleted), the files already there (no download), a file
whose pinned sha256 does not match (counted, the archive kept), an archive whose sha256 does not match (nothing taken
out).
"""
import hashlib
import io
import json
import pathlib
import sys
import tarfile
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "pc"))
import fetch  # noqa: E402


def sha(b):
    return hashlib.sha256(b).hexdigest()


def make_archive(path, members):
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members:
            ti = tarfile.TarInfo(name)
            if data is None:
                ti.type = tarfile.DIRTYPE
                tar.addfile(ti)
            else:
                ti.size = len(data)
                tar.addfile(ti, io.BytesIO(data))


def source(arc, files, arc_sha=None):
    b = arc.read_bytes()
    return {"archive": arc.as_uri(), "archive_bytes": len(b), "archive_sha256": arc_sha or sha(b), "dest": "fixtures/t",
            "for": "m", "bytes": len(b), "files": files}


def main():
    x, y = b"x" * 44, b"yy" * 30
    cases = []
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        arc = d / "src" / "t.tar.gz"
        arc.parent.mkdir()
        make_archive(arc, [("./", None), ("./README.md", b"r"), ("./a/", None), ("./a/x.wav", x), ("./b/y.wav", y),
                           ("./b/z.wav", b"not asked for")])
        files = [{"path": "a/x.wav", "size": len(x), "sha256": sha(x)}, {"path": "b/y.wav", "size": len(y), "sha256": sha(y)}]

        root = d / "root1"
        rep = []
        bad = fetch.fetch_archive(source(arc, files), root, rep)
        got = sorted(str(p.relative_to(root / "fixtures/t")) for p in (root / "fixtures/t").rglob("*") if p.is_file())
        cases.append(("fresh folder", bad == 0 and got == ["a/x.wav", "b/y.wav"] and all(r["ok"] for r in rep)
                      and rep[0]["tries"] == 1 and not (root / "downloads" / "t.tar.gz").exists()))

        rep = []
        bad = fetch.fetch_archive(source(arc, files), root, rep)
        cases.append(("files already there", bad == 0 and all(r["tries"] == 0 and r["ok"] for r in rep)))

        root = d / "root2"
        rep = []
        wrong = [files[0], dict(files[1], sha256="0" * 64)]
        bad = fetch.fetch_archive(source(arc, wrong), root, rep)
        cases.append(("a file's sha256 differs", bad == 1 and [r["ok"] for r in rep] == [True, False]
                      and (root / "downloads" / "t.tar.gz").exists()))

        root = d / "root3"
        rep = []
        bad = fetch.fetch_archive(source(arc, files, arc_sha="0" * 64), root, rep)
        cases.append(("the archive's sha256 differs", bad == 2 and not (root / "fixtures/t").exists()))

    for name, good in cases:
        print(f"{'ok  ' if good else 'FAIL'} {name}")
    n = sum(g for _, g in cases)
    print(f"test_fetch_archive: {n} of {len(cases)} cases as expected")
    return 0 if n == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
