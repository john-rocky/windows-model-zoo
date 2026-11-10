"""A Foundry Local LLM cell: the package files of a Foundry variant, registered as a local model whose chat template
returns the message content as it is, then the reference's rendered prompts through the daemon's REST; one JSON.

    python run_llm_foundry.py --package <folder> --id z6-<name>:1 --device cpu|gpu [--ep <provider>]
        [--task chat-completion|vision-language-chat] --work <folder> --cuda12 <dll folder> [--daemon-env K=V ...]
        --ref <ref.json> --out run.json [--max-new 48] [--gate-max-new 16]
    python run_llm_foundry.py --cleanup --id z6-<name>:1 --cuda12 <dll folder>   # pc/chain.ps1 after a killed run

Foundry Local's REST has no raw-prompt route (/v1/completions = 404) and its chat wraps the messages with the folder's
chat_template.jinja (PLAN.md section Z6, the probes of 11-10 07:54-07:56). So: <work>\\<name> = hard links of the
package's files except chat_template.jinja (the template {%- for message in messages -%}{{- message['content'] -}}
{%- endfor -%}), tokenizer_config.json (its chat_template, when it has one, the same), inference_model.json (this id)
and genai_config.json (copied; search do_sample false and repetition_penalty 1.0 when the package sets otherwise:
plain greedy); `foundry model register <dir> <id> --task ... --device ... [--execution-provider ...]`, `model load`,
then POST /catalogs/local/v1/chat/completions {messages: [one user message = the rendered string], max_tokens 48 (the
gate prompt 16), temperature 0}; then unload, unregister, `server stop`. The daemon (one per Windows user) is this
cell's: the cell does not start while a note of another lane that shares the daemon (foundry_notes in
pcup\\zoo_yield.json) is in active\\ or a daemon it did not start runs; CUDA 12's DLL folder goes on the daemon's PATH
(the shared PC's rule).
The start token (PLAN.md section Z6, change 1): the untimed warm-up request (the first prompt, 8 tokens) reads the
daemon's prompt token count; a rendered string that starts with <bos> and comes back one token over the reference's
ids goes without that <bos> in every scored request (the tokenizer adds one). A count other than the reference's (or
one over it for a <bos> string) means the registration wraps the prompt: the cell is an error.
A row: the daemon's prompt / completion token counts (no ids here), the text, and whether it ended on an end token
(finish_reason stop).
"""
import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zoolib as al  # noqa: E402

IDENTITY = "{%- for message in messages -%}{{- message['content'] -}}{%- endfor -%}"
HOME = pathlib.Path(os.environ.get("USERPROFILE") or os.path.expanduser("~"))
NOTES = HOME / "pcup" / "active"


def foreign_notes():
    """other lanes' Foundry notes (another lane may share the user's one daemon, PLAN.md section Z6): the notes in
    pcup\\active that match foundry_notes in pcup\\zoo_yield.json"""
    return al.yield_notes("foundry_notes")


def daemon_pids():
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq foundrylocald.exe", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return []
    return [int(x.split('","')[1]) for x in out.strip().splitlines() if x.startswith('"foundrylocald')]


def foundry(log, *parts, env=None, timeout=900):
    exe = shutil.which("foundry") or "foundry"
    t = time.perf_counter()
    p = subprocess.run([exe, *parts], capture_output=True, timeout=timeout, env=env, stdin=subprocess.DEVNULL)
    text = (p.stdout + p.stderr).decode("utf-8", "replace")
    row = {"command": "foundry " + " ".join(parts), "exit": p.returncode, "s": round(time.perf_counter() - t, 2),
           "tail": text.strip().splitlines()[-6:]}
    log.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def stop_daemon(log, env):
    foundry(log, "server", "stop", env=env, timeout=120)
    for _ in range(30):
        if not daemon_pids():
            return "stopped"
        time.sleep(1)
    subprocess.run(["taskkill", "/IM", "foundrylocald.exe", "/F"], capture_output=True)
    time.sleep(2)
    return "killed"


def link_package(src, dst, ident, rec):
    """dst = hard links of src's files, the four files above written instead"""
    if dst.exists():
        shutil.rmtree(dst)
    changed = {}
    for f in sorted(p for p in src.rglob("*") if p.is_file()):
        rel = f.relative_to(src)
        t = dst / rel
        t.parent.mkdir(parents=True, exist_ok=True)
        name = rel.as_posix()
        if name == "chat_template.jinja":
            t.write_text(IDENTITY, encoding="utf-8")
            changed[name] = "the identity template"
        elif name == "tokenizer_config.json":
            j = json.loads(f.read_text(encoding="utf-8"))
            if "chat_template" in j:
                j["chat_template"] = IDENTITY
                t.write_text(json.dumps(j, indent=1, ensure_ascii=False), encoding="utf-8")
                changed[name] = "chat_template = the identity template"
            else:
                os.link(f, t)
        elif name == "inference_model.json":
            t.write_text(json.dumps({"Name": ident, "PromptTemplate": {"system": "{Content}", "user": "{Content}",
                                                                       "assistant": "{Content}", "prompt": "{Content}"}}),
                         encoding="utf-8")
            changed[name] = f"Name {ident}, PromptTemplate {{Content}} (Foundry reads chat_template.jinja first)"
        elif name == "genai_config.json":
            j = json.loads(f.read_text(encoding="utf-8"))
            s = j.setdefault("search", {})
            rec["genai_config_search_package"] = dict(s)
            if s.get("do_sample") is not False or s.get("repetition_penalty", 1.0) != 1.0:
                s["do_sample"] = False
                s["repetition_penalty"] = 1.0
                changed[name] = "search do_sample false, repetition_penalty 1.0 (plain greedy)"
            t.write_text(json.dumps(j, indent=1), encoding="utf-8")
            rec["genai_config_search_used"] = dict(s)
        else:
            os.link(f, t)
    if "chat_template.jinja" not in changed:   # a package without one: the identity template goes in anyway
        (dst / "chat_template.jinja").write_text(IDENTITY, encoding="utf-8")
        changed["chat_template.jinja"] = "added: the identity template"
    return changed


def main():
    al.utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default="")
    ap.add_argument("--id", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ep", default="")
    ap.add_argument("--task", default="chat-completion")
    ap.add_argument("--work", default="")
    ap.add_argument("--cuda12", required=True)
    ap.add_argument("--daemon-env", action="append", default=[])
    ap.add_argument("--ref", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--gate-max-new", type=int, default=16)
    ap.add_argument("--cleanup", action="store_true")
    args = ap.parse_args()
    env = dict(os.environ)
    env["PATH"] = args.cuda12 + os.pathsep + env.get("PATH", "")
    if args.cleanup:
        log = []
        foundry(log, "model", "unload", args.id, env=env, timeout=120)
        foundry(log, "model", "unregister", args.id, env=env, timeout=120)
        notes = foreign_notes()
        how = f"daemon left running: other lanes' Foundry notes {notes}" if notes else stop_daemon(log, env)
        print(json.dumps({"cleanup": "foundry", "id": args.id, "daemon": how}), flush=True)
        return 0
    rec = al.Record("run_llm_foundry.py", args)
    al.banner()
    code = 0
    started = False
    log = []
    rec["foundry_cli"] = log
    try:
        rec["versions"] = al.versions([])
        notes = foreign_notes()
        if notes:
            raise RuntimeError(f"not run: other lanes' Foundry notes {notes} (the user's one daemon)")
        pids = daemon_pids()
        if pids:
            raise RuntimeError(f"not run: a Foundry daemon this cell did not start runs (pid {pids})")
        ref = json.loads(pathlib.Path(args.ref).read_text(encoding="utf-8"))
        rows = [r for r in ref["rows"] if r.get("prompt")]
        src = pathlib.Path(args.package)
        dst = pathlib.Path(args.work) / args.id.split(":")[0]
        rec["package"] = {"src": str(src), "dir": str(dst), "changed": link_package(src, dst, args.id, rec)}
        denv = dict(env)
        for kv in args.daemon_env:
            k, v = kv.split("=", 1)
            denv[k] = v
        r = foundry(log, "--version", env=env, timeout=60)
        rec["foundry_version"] = (r["tail"] or [None])[-1]
        r = foundry(log, "server", "start", env=denv, timeout=300)
        started = True
        if r["exit"] != 0:
            raise RuntimeError(f"foundry server start exit {r['exit']}: {r['tail']}")
        rec["daemon_pids"] = daemon_pids()
        foundry(log, "model", "unregister", args.id, env=env, timeout=120)
        reg = ["model", "register", str(dst), args.id, "--task", args.task, "--device", args.device,
               "--supports-tool-calling", "false", "--supports-reasoning", "false"]
        if args.ep:
            reg += ["--execution-provider", args.ep]
        r = foundry(log, *reg, env=env, timeout=300)
        if r["exit"] != 0:
            raise RuntimeError(f"foundry model register exit {r['exit']}: {r['tail']}")
        t = time.perf_counter()
        r = foundry(log, "model", "load", args.id, env=env, timeout=1800)
        rec["load_s"] = round(time.perf_counter() - t, 3)
        if r["exit"] != 0:
            raise RuntimeError(f"foundry model load exit {r['exit']}: {r['tail']}")
        d = json.loads((HOME / ".foundry" / "daemon.json").read_text(encoding="utf-8"))
        url = (d.get("web_urls") or d.get("webUrls"))[0].rstrip("/") + "/catalogs/local/v1/chat/completions"
        rec["url"] = url

        def chat(text, mx):
            body = {"model": args.id, "messages": [{"role": "user", "content": text}], "max_tokens": mx,
                    "temperature": 0, "stream": False}
            req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json"}, method="POST")
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=1800) as resp:
                    j = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                raise RuntimeError(f"HTTP {e.code} {e.read().decode('utf-8', 'replace')[:400]}")
            return j, time.perf_counter() - t0

        first = rows[0]
        n_ref = len(first.get("prompt_ids") or [])
        j, ws = chat(first["prompt"], 8)
        rec["warmup_s"] = round(ws, 3)
        pt = (j.get("usage") or {}).get("prompt_tokens")
        drop_bos = False
        if pt == n_ref:
            rule = "as it is (the warm-up's prompt tokens = the reference's ids)"
        elif pt == n_ref + 1 and first["prompt"].startswith("<bos>"):
            drop_bos = True
            rule = "without the leading <bos> (the warm-up's prompt tokens = the reference's ids + 1)"
        else:
            raise RuntimeError(f"the registration wraps the prompt: warm-up prompt_tokens {pt}, reference ids {n_ref}")
        rec["bos_rule"] = {"warmup_prompt_tokens": pt, "reference_ids": n_ref, "sent": rule}
        rec["workload"] = {"ref": args.ref, "prompts": len(rows), "id": args.id, "task": args.task,
                           "device": args.device, "ep": args.ep or None, "daemon_env": args.daemon_env,
                           "decode": "greedy (temperature 0; genai_config search do_sample false, repetition_penalty 1.0)",
                           "max_new_tokens": args.max_new, "gate_max_new_tokens": args.gate_max_new,
                           "chat_template": "the identity template: one user message = the reference's rendered prompt, "
                                            + rule}
        rec.total = len(rows)
        for r in rows:
            mx = args.gate_max_new if r.get("role") == "gate" else args.max_new
            text = r["prompt"]
            if drop_bos and text.startswith("<bos>"):
                text = text[len("<bos>"):]
            j, wall = chat(text, mx)
            u = j.get("usage") or {}
            c = (j.get("choices") or [{}])[0]
            fin = c.get("finish_reason")
            rec.row({"id": r["id"], "role": r.get("role"), "prompt_tokens": u.get("prompt_tokens"),
                     "prompt_form": "one user message through the identity template, " + rule,
                     "new_tokens": u.get("completion_tokens"), "text": (c.get("message") or {}).get("content"),
                     "eos": fin == "stop", "stop": fin, "max_new": mx, "wall_s": wall})
        rec["summary"] = rec.summarize("llm")
        rec["status"] = "OK"
    except Exception:
        rec.fail()
        code = 1
    finally:
        if started:
            try:
                foundry(log, "model", "unload", args.id, env=env, timeout=120)
                foundry(log, "model", "unregister", args.id, env=env, timeout=120)
                rec["daemon_stop"] = stop_daemon(log, env)
            except Exception as e:
                rec["daemon_stop"] = f"failed: {e!r}"
    rec.write()
    return code


if __name__ == "__main__":
    sys.exit(main())
