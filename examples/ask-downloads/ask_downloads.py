"""Ask the same question to the original model and to downloads of it, side by side, on one Windows PC.

    python ask_downloads.py --scene scene_npu.json [--root <work folder>]

One window: a question box and one card per engine. The first card is the original model (transformers on the CPU,
fp32); the others are downloads of it, each in its own runtime and device. A card shows the download's name, runtime
and device, its result in windows-model-zoo (the qa20 line of its task cell: pass or fail, and how many of 20 short
answers' first lines equal the original model's), and the answer it gives now. Each engine runs in its own process
(worker.py) with the Python of its own venv.

The question is wrapped once by the original model's chat template and the same string goes to every card, decoded
greedily, the way the zoo's qa20 runners do it: so for one of the zoo's 20 questions a card shows the answer the
zoo's record holds, and for your own question it shows what that download answers on this PC. A download's answer is
compared with the original's the way the qa20 line compares them: the first line, lower case, quotes, backticks or
asterisks around it and a final . ! ? dropped, spaces collapsed.

The scene file (JSON) names the cards; paths in it may start with {ROOT} (--root, default: this folder's "work"),
{ZOO} (the zoo repo: two folders up) or {HERE} (this folder). For recording, --autoplay types the zoo's questions
(ids of suites/llm/qa20.jsonl) and presses Ask after the go file appears; it moves the mouse pointer to what it
types into and presses. --run-dir gets events.jsonl, answers.jsonl, ready.json and finished.json.

A card whose model does not load shows the error, and the other cards go on. The first card renders every card's
prompt, so when it fails nothing more is asked (the other cards say so).
"""
import argparse
import ctypes
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk

HERE = os.path.dirname(os.path.abspath(__file__))
CREATE_NO_WINDOW = 0x08000000
COLORS = {"bg": "#F3F3F3", "card": "#FFFFFF", "line": "#D9D9D9", "text": "#1A1A1A", "dim": "#5F5F5F",
          "accent": "#0067C0", "accent_dn": "#004E93", "pass": "#0F7B0F", "fail": "#C42B1C", "ref": "#6B6B6B"}
NO_PROMPT = "not asked: every card's prompt comes from the original model's card"


def norm_answer(t):
    """The qa20 line's comparison form (windows-model-zoo tools/results.py norm_answer)."""
    lines = (t or "").strip().splitlines()
    s = lines[0] if lines else ""
    prev = None
    while s != prev:
        prev = s
        s = s.strip().lower().strip("\"'`*").rstrip(".!?")
    return " ".join(s.split())


def first_line(t):
    lines = (t or "").strip().splitlines()
    return lines[0].strip() if lines else ""


def short_error(err):
    """A card's line for a worker's error: the first line's kind ("load: RuntimeError: ") and the last line, where
    OpenVINO puts the reason after its "Exception from <source file>" lines. The whole text stays in events.jsonl."""
    lines = [x.strip() for x in (err or "").splitlines() if x.strip()]
    if len(lines) <= 1:
        return err or ""
    head = re.match(r"(?:load: )?\w+: ", lines[0])
    return (head.group(0) if head else "") + lines[-1]


def dpi_aware():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)   # per monitor: Tk draws in physical pixels
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


class Log:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.f = self.a = None
        if run_dir:
            os.makedirs(run_dir, exist_ok=True)
            self.f = open(os.path.join(run_dir, "events.jsonl"), "a", encoding="utf-8")
            self.a = open(os.path.join(run_dir, "answers.jsonl"), "a", encoding="utf-8")

    def ev(self, kind, **kw):
        if self.f:
            row = {"t": time.time(), "perf": time.perf_counter(), "kind": kind}
            row.update(kw)
            self.f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.f.flush()

    def answer(self, row):
        if self.a:
            self.a.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.a.flush()

    def write(self, name, obj):
        if self.run_dir:
            p = os.path.join(self.run_dir, name)
            with open(p + ".tmp", "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=1)
            os.replace(p + ".tmp", p)


class Worker:
    """worker.py in its own process; its stdout JSON lines go to a queue the Tk loop reads."""

    def __init__(self, idx, spec, resolve, log_dir):
        self.idx, self.spec = idx, spec
        w = spec["worker"]
        py = resolve(w["python"])
        args = [py, os.path.join(HERE, "worker.py"), "--kind", w["kind"]]
        for k in ("model_dir", "device", "cache_dir", "server", "model"):
            if w.get(k):
                args += ["--" + k.replace("_", "-"), resolve(w[k])]
        self.args = args
        self.q = queue.Queue()
        err = open(os.path.join(log_dir, f"worker{idx}.stderr.txt"), "ab") if log_dir else subprocess.DEVNULL
        env = dict(os.environ)
        env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "ORT_DISABLE_TELEMETRY": "1"})
        env.update({k: resolve(v) for k, v in (w.get("env") or {}).items()})
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, cwd=HERE,
                                  env=env, creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for raw in self.p.stdout:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                self.q.put(json.loads(line))
            except ValueError:
                self.q.put({"event": "noise", "line": line[:300]})
        self.q.put({"event": "exited", "code": self.p.wait()})

    def send(self, obj):
        try:
            self.p.stdin.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
            self.p.stdin.flush()
            return True
        except Exception:
            return False

    def stop(self):
        self.send({"op": "quit"})
        try:
            self.p.wait(timeout=8)
        except Exception:
            pass
        if self.p.poll() is None and os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(self.p.pid), "/T", "/F"], capture_output=True)
        elif self.p.poll() is None:
            self.p.kill()


class Card:
    def __init__(self, app, parent, idx, spec, zoo):
        self.app, self.idx, self.spec, self.zoo = app, idx, spec, zoo
        s, C = app.s, COLORS
        f = tk.Frame(parent, bg=C["card"], highlightthickness=s(2), highlightbackground=C["line"])
        self.frame = f
        pad = s(36)
        tk.Label(f, text=spec["role"].upper(), font=app.font(26, "bold"), fg=C["dim"], bg=C["card"],
                 anchor="w").pack(fill="x", padx=pad, pady=(s(30), 0))
        tk.Label(f, text=spec["name"], font=app.font(app.name_px, "bold"), fg=C["text"], bg=C["card"], anchor="w",
                 justify="left", wraplength=app.card_w - 2 * pad).pack(fill="x", padx=pad, pady=(s(10), 0))
        tk.Label(f, text=spec["engine"], font=app.font(30), fg=C["dim"], bg=C["card"], anchor="w", justify="left",
                 wraplength=app.card_w - 2 * pad).pack(fill="x", padx=pad, pady=(s(6), 0))
        if zoo is None or zoo["pass"] is None:   # the original model's card, or a download without a zoo result
            note = spec.get("note") or ("The zoo compares every download with this model" if idx == 0
                                        else "Not in windows-model-zoo yet: no result for this download on this device")
            box = tk.Frame(f, bg="#EDEDED")
            box.pack(fill="x", padx=pad, pady=(s(26), 0))
            tk.Label(box, text=note, font=app.font(28), fg=C["ref"], bg="#EDEDED", anchor="w", justify="left",
                     wraplength=app.card_w - 2 * pad - s(40)).pack(fill="x", padx=s(20), pady=s(18))
            app.log.ev("ui", where=f"card{idx}.zoo", text=note)
        else:
            col = C["pass"] if zoo["pass"] else C["fail"]
            box = tk.Frame(f, bg=col)
            box.pack(fill="x", padx=pad, pady=(s(26), 0))
            tk.Label(box, text="windows-model-zoo: " + ("PASS" if zoo["pass"] else "FAIL"), font=app.font(36, "bold"),
                     fg="#FFFFFF", bg=col, anchor="w").pack(fill="x", padx=s(20), pady=(s(14), 0))
            tk.Label(box, text=f"{zoo['agree']} of {zoo['n']} answers match the original (pass mark {zoo['bar']})",
                     font=app.font(27), fg="#FFFFFF", bg=col, anchor="w", justify="left",
                     wraplength=app.card_w - 2 * pad - s(40)).pack(fill="x", padx=s(20), pady=(s(4), s(16)))
            app.log.ev("ui", where=f"card{idx}.zoo", text=f"{'PASS' if zoo['pass'] else 'FAIL'} {zoo['agree']}/{zoo['n']}",
                       source=zoo["source"])
        tk.Frame(f, bg=C["line"], height=s(2)).pack(fill="x", padx=pad, pady=(s(30), 0))
        self.state = tk.Label(f, text="loading the model…", font=app.font(28), fg=C["dim"], bg=C["card"], anchor="w",
                              justify="left", wraplength=app.card_w - 2 * pad)   # a long error wraps inside the card
        self.state.pack(fill="x", padx=pad, pady=(s(20), 0))
        box = tk.Frame(f, bg=C["card"], height=s(240))   # a fixed height: an answer never moves the lines below it
        box.pack(fill="x", padx=pad, pady=(s(6), 0))
        box.pack_propagate(False)
        self.answer = tk.Label(box, text="", font=app.font(170, "bold"), fg=C["text"], bg=C["card"], anchor="w",
                               justify="left", wraplength=app.card_w - 2 * pad)
        self.answer.pack(fill="both", expand=True)
        self.mark = tk.Label(f, text=" ", font=app.font(46, "bold", family="Segoe UI Symbol"), fg=C["dim"],
                             bg=C["card"], anchor="w")
        self.mark.pack(fill="x", padx=pad, pady=(s(10), 0))
        self.more = tk.Label(f, text="", font=app.font(24), fg=C["dim"], bg=C["card"], anchor="nw", justify="left",
                             wraplength=app.card_w - 2 * pad)
        self.more.pack(fill="both", expand=True, padx=pad, pady=(s(10), s(24)))
        self.text = None

    def set_state(self, text):
        self.state.config(text=text)
        self.app.log.ev("ui", where=f"card{self.idx}.state", text=text)

    def clear(self):
        self.text = None
        self.answer.config(text="")
        self.mark.config(text=" ")
        self.more.config(text="")

    def show(self, text):
        self.text = text
        head = first_line(text) or "(empty)"
        n = len(head)
        size = 170 if n <= 6 else 130 if n <= 10 else 90 if n <= 16 else 60 if n <= 30 else 44 if n <= 60 else 32
        self.answer.config(text=head, font=self.app.font(size, "bold"), fg=COLORS["text"])
        rest = (text or "").strip()[len(head):].strip()
        self.more.config(text=rest[:400])
        self.app.log.ev("ui", where=f"card{self.idx}.answer", text=head)

    def compare(self, original):
        if self.idx == 0 or self.text is None or original is None:
            return
        same = norm_answer(self.text) == norm_answer(original)
        self.mark.config(text=("✓ same as the original" if same else "✗ differs from the original"),
                         fg=COLORS["pass"] if same else COLORS["fail"])
        self.answer.config(fg=COLORS["pass"] if same else COLORS["fail"])
        self.app.log.ev("ui", where=f"card{self.idx}.mark", text="same" if same else "differs")


class App:
    def __init__(self, a):
        self.a = a
        self.scene_path = os.path.abspath(a.scene)
        with open(self.scene_path, encoding="utf-8") as f:
            self.scene = json.load(f)
        self.root_dir = os.path.abspath(a.root or os.path.join(HERE, "work"))
        self.zoo_dir = os.path.abspath(os.path.join(HERE, "..", ".."))
        self.log = Log(a.run_dir)
        self.questions = self.load_qa20()
        self.zoo = [self.zoo_line(c) for c in self.scene["cards"]]

        dpi_aware()
        self.tk = tk.Tk()
        self.tk.title(self.scene.get("window_title", "Ask the downloads"))
        self.tk.configure(bg=COLORS["bg"])
        sw = self.tk.winfo_screenwidth()
        self.scale = sw / 2560.0
        n = len(self.scene["cards"])
        self.margin = self.s(40)
        self.card_w = int((sw - self.margin * (n + 1)) / n)
        # every card's name on one line, at one size (a repo name has no space to wrap at)
        self.name_px = self.fit_px([c["name"] for c in self.scene["cards"]], 34, self.card_w - 2 * self.s(36) - self.s(8))
        self.build()
        self.tk.state("zoomed")
        self.tk.protocol("WM_DELETE_WINDOW", self.quit)
        if a.autoplay:   # recording: in front of whatever window was there (a task's window may not get the focus)
            self.tk.attributes("-topmost", True)
            self.tk.lift()

        os.makedirs(self.root_dir, exist_ok=True)
        log_dir = a.run_dir or None
        self.workers = []
        for i, c in enumerate(self.scene["cards"]):
            try:
                self.workers.append(Worker(i, c, self.resolve, log_dir))
            except Exception as e:
                self.workers.append(None)
                self.cards[i].set_state(f"could not start: {type(e).__name__}: {e}")
        self.loaded = [False] * n
        self.warm = [False] * n
        self.failed = [w is None for w in self.workers]   # a worker that could not start has failed from the start
        self.info = [None] * n
        self.pending = {}          # request id -> (kind, card index or None, question)
        self.next_id = 1
        self.ready = False
        self.busy = False
        self.turn = None          # {"qid", "text", "answers": {i: text}, "t0"}
        self.autoplay = [q.strip() for q in (a.autoplay or "").split(",") if q.strip()]
        self.finished = False
        self.tk.after(50, self.poll)
        self.tk.after(60, self.check_ready)   # the first card may have failed already (could not start)

    # --- helpers --------------------------------------------------------------------------------------------------
    def s(self, px):
        return max(1, int(round(px * getattr(self, "scale", 1.0))))

    def font(self, px, weight="normal", family=None):
        return (family or self.scene.get("font", "Segoe UI"), -self.s(px), weight)

    def fit_px(self, texts, px, width, weight="bold"):
        """The largest size up to px (in 2560-wide pixels) at which every text fits width (screen pixels) on one line."""
        import tkinter.font as tkfont
        while px > 18:
            f = tkfont.Font(root=self.tk, family=self.scene.get("font", "Segoe UI"), size=-self.s(px), weight=weight)
            if all(f.measure(t) <= width for t in texts):
                break
            px -= 1
        return px

    def resolve(self, p):
        return (str(p).replace("{ROOT}", self.root_dir).replace("{ZOO}", self.zoo_dir).replace("{HERE}", HERE))

    def load_qa20(self):
        p = self.resolve(self.scene.get("questions", "{ZOO}/suites/llm/qa20.jsonl"))
        out = {}
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        out[r["id"]] = r["text"]
        except OSError:
            pass
        return out

    def zoo_line(self, card):
        p = card.get("zoo_cell")
        if not p:
            return None
        path = self.resolve(p)
        try:
            with open(path, encoding="utf-8") as f:
                cell = json.load(f)
            line = cell["line"]
            ans = line["detail"]["answers"]
            return {"pass": bool(line["pass"]), "agree": int(ans["agree"]), "n": int(ans["n"]), "bar": int(ans["bar"]),
                    "source": os.path.relpath(path, self.zoo_dir) if path.startswith(self.zoo_dir) else path}
        except (OSError, KeyError, TypeError, ValueError):   # a download the zoo has not checked (or no such file)
            return {"pass": None, "source": path}

    # --- UI -------------------------------------------------------------------------------------------------------
    def build(self):
        C, s = COLORS, self.s
        top = tk.Frame(self.tk, bg=C["bg"])
        top.pack(fill="x", padx=self.margin, pady=(s(30), 0))
        tk.Label(top, text=self.scene["title"], font=self.font(52, "bold"), fg=C["text"], bg=C["bg"],
                 anchor="w").pack(side="left")
        tk.Label(top, text=self.scene.get("subtitle", ""), font=self.font(30), fg=C["dim"], bg=C["bg"],
                 anchor="w").pack(side="left", padx=(s(28), 0), pady=(s(14), 0))
        row = tk.Frame(self.tk, bg=C["bg"])
        row.pack(fill="x", padx=self.margin, pady=(s(26), 0))
        self.entry = tk.Entry(row, font=self.font(44), relief="flat", bg="#FFFFFF", fg=C["text"],
                              insertbackground=C["text"], highlightthickness=s(2), highlightbackground="#8A8A8A",
                              highlightcolor=C["accent"])
        self.entry.pack(side="left", fill="x", expand=True, ipady=s(18))
        self.entry.bind("<Return>", lambda e: self.ask())
        self.button = tk.Label(row, text="Ask", font=self.font(42, "bold"), fg="#FFFFFF", bg="#9BB8D3",
                               padx=s(60), pady=s(18), cursor="hand2")
        self.button.pack(side="left", padx=(s(24), 0))
        self.button.bind("<Button-1>", lambda e: self.ask())
        grid = tk.Frame(self.tk, bg=C["bg"])
        grid.pack(fill="both", expand=True, padx=self.margin, pady=(s(30), s(30)))
        self.cards = []
        for i, c in enumerate(self.scene["cards"]):
            card = Card(self, grid, i, c, self.zoo[i])
            card.frame.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else self.margin // 2,
                                                                   0 if i == len(self.scene["cards"]) - 1 else self.margin // 2))
            grid.columnconfigure(i, weight=1, uniform="cards")
            self.cards.append(card)
        grid.rowconfigure(0, weight=1)

    def set_ready(self, on):
        self.button.config(bg=COLORS["accent"] if on else "#9BB8D3")

    def rect(self, w):
        w.update_idletasks()
        return [w.winfo_rootx(), w.winfo_rooty(), w.winfo_rootx() + w.winfo_width(), w.winfo_rooty() + w.winfo_height()]

    # --- workers --------------------------------------------------------------------------------------------------
    def req(self, i, op, kind_card, question=None, **kw):
        rid = self.next_id
        self.next_id += 1
        self.pending[rid] = (op, kind_card, question)
        body = {"op": op, "id": rid}
        body.update(kw)
        if not self.workers[i] or not self.workers[i].send(body):
            self.on_error(i, rid, "the worker is not running")
        return rid

    def poll(self):
        for i, w in enumerate(self.workers):
            if not w:
                continue
            while True:
                try:
                    m = w.q.get_nowait()
                except queue.Empty:
                    break
                self.on_msg(i, m)
        if not self.finished:
            self.tk.after(30, self.poll)

    def on_msg(self, i, m):
        ev = m.get("event")
        rid = m.get("id")
        what = self.pending.pop(rid, None) if rid is not None else None
        if ev == "ready":
            self.loaded[i], self.info[i] = True, {"load_s": m.get("load_s"), **(m.get("info") or {})}
            self.log.ev("worker_ready", card=i, load_s=m.get("load_s"), info=m.get("info"))
            self.cards[i].set_state(NO_PROMPT if i != 0 and self.failed[0] else "warming up…")
            if i == 0:
                q = self.questions.get("q01", "What is the capital of France? Answer with one word.")
                self.req(0, "render", None, question=("warmup", q), text=q)
        elif ev == "rendered":
            kind, q = what[2] if what else (None, None)
            if kind == "warmup":
                self.warm_prompt = m["prompt"]
                for j in range(len(self.workers)):
                    if self.loaded[j]:
                        self.req(j, "warmup", j, prompt=self.warm_prompt)
            elif kind == "ask" and self.turn is not None:
                self.dispatch(m["prompt"])
        elif ev == "warm":
            self.warm[i] = True
            self.cards[i].set_state("ready")
            self.log.ev("worker_warm", card=i, s=m.get("s"))
            self.check_ready()
        elif ev == "answer":
            self.on_answer(i, m, what)
        elif ev == "error":
            self.on_error(i, rid, m.get("error", "error"), m.get("trace"))
        elif ev == "exited":
            if not self.finished and not self.failed[i]:   # after an error the card keeps that error's text
                self.on_error(i, None, f"the worker exited ({m.get('code')})")
        elif ev == "noise":
            self.log.ev("worker_noise", card=i, line=m.get("line"))
        # a worker that became ready after the warm-up prompt was rendered
        if ev == "ready" and i != 0 and getattr(self, "warm_prompt", None):
            self.req(i, "warmup", i, prompt=self.warm_prompt)

    def on_error(self, i, rid, err, trace=None):
        self.failed[i] = True
        self.cards[i].set_state("error: " + short_error(err)[:200])
        self.log.ev("worker_error", card=i, id=rid, error=err, trace=trace)
        if i == 0:
            self.no_prompt()
        if self.busy and i not in self.turn["answers"]:
            self.turn["answers"][i] = None
            self.after_answer()
        self.check_ready()

    def no_prompt(self):
        """The first card failed. It renders every card's prompt, so nothing more is asked; a question already sent
        to the other cards still gets their answers."""
        self.set_ready(False)
        if self.busy and "prompt" in self.turn:
            return
        for j in range(1, len(self.cards)):
            if self.busy:   # its prompt never came: no other card was asked
                self.turn["answers"].setdefault(j, None)
            if not self.failed[j]:
                self.cards[j].set_state(NO_PROMPT)

    def check_ready(self):
        if self.ready:
            return
        if self.failed[0]:   # never ready: said once (ready.json with ok false; an autoplay run ends)
            if not getattr(self, "gave_up", False):
                self.gave_up = True
                self.no_prompt()
                self.write_ready()
                if self.autoplay:
                    self.finish(ok=False)
            return
        if all(self.warm[j] or self.failed[j] for j in range(len(self.workers))):
            self.ready = True
            self.set_ready(True)
            self.write_ready()
            if self.autoplay:
                self.wait_go()

    def write_ready(self):
        self.tk.update_idletasks()
        geo = {"window": self.rect(self.tk), "entry": self.rect(self.entry), "button": self.rect(self.button),
               "cards": [self.rect(c.frame) for c in self.cards], "answers": [self.rect(c.answer) for c in self.cards],
               "marks": [self.rect(c.mark) for c in self.cards], "screen": [self.tk.winfo_screenwidth(),
                                                                           self.tk.winfo_screenheight()],
               "zoomed": self.tk.state() == "zoomed", "workers_ok": [self.warm[j] for j in range(len(self.workers))],
               "info": self.info, "zoo": self.zoo, "scene": os.path.basename(self.scene_path), "t": time.time()}
        geo["ok"] = all(geo["workers_ok"])
        self.log.write("ready.json", geo)
        self.log.ev("ready", ok=geo["ok"])

    # --- asking ---------------------------------------------------------------------------------------------------
    def ask(self, qid=None):
        if not self.ready or self.busy or self.failed[0]:
            return
        text = self.entry.get().strip()
        if not text:
            return
        self.busy = True
        self.set_ready(False)
        self.turn = {"qid": qid, "text": text, "answers": {}, "t0": time.perf_counter()}
        for c in self.cards:
            c.clear()
            if not self.failed[c.idx]:
                c.set_state("answering…")
        self.log.ev("ask", qid=qid, text=text)
        self.req(0, "render", None, question=("ask", text), text=text)

    def dispatch(self, prompt):
        self.turn["prompt"] = prompt
        for j in range(len(self.workers)):
            if not self.failed[j]:
                self.req(j, "ask", j, question=self.turn["text"], prompt=prompt)
            else:
                self.turn["answers"][j] = None
        self.after_answer()

    def on_answer(self, i, m, what):
        if self.turn is None:
            return
        text = m.get("text", "")
        self.turn["answers"][i] = text
        c = self.cards[i]
        c.set_state("Answer")
        c.show(text)
        self.log.answer({"t": time.time(), "qid": self.turn["qid"], "question": self.turn["text"], "card": i,
                         "name": self.scene["cards"][i]["name"], "text": text, "new_ids": m.get("new_ids"),
                         "prompt_ids": m.get("prompt_ids"), "s": m.get("s")})
        orig = self.turn["answers"].get(0)
        if i == 0:
            for j in range(1, len(self.cards)):
                if self.turn["answers"].get(j) is not None:
                    self.cards[j].compare(text)
        elif orig is not None:
            c.compare(orig)
        self.after_answer()

    def after_answer(self):
        if self.turn is None or len(self.turn["answers"]) < len(self.cards):
            return
        self.log.ev("turn_done", qid=self.turn["qid"], answers=self.turn["answers"])
        self.busy = False
        self.set_ready(not self.failed[0])
        self.done_turns = getattr(self, "done_turns", []) + [dict(self.turn)]
        if self.autoplay_queue is not None:
            self.tk.after(self.a.hold_ms, self.play_next)

    # --- autoplay (recording only) --------------------------------------------------------------------------------
    autoplay_queue = None

    def wait_go(self):
        if self.a.go_file and not os.path.exists(self.a.go_file):
            self.tk.after(50, self.wait_go)
            return
        self.log.ev("go")
        self.autoplay_queue = list(self.autoplay)
        self.tk.after(self.a.lead_ms, self.play_next)

    def play_next(self):
        if self.failed[0]:   # the first card failed during the run: nothing more can be asked
            self.finish(ok=False)
            return
        if not self.autoplay_queue:
            self.tk.after(self.a.end_ms, self.finish)
            return
        qid = self.autoplay_queue.pop(0)
        text = self.questions.get(qid)
        if not text:
            self.log.ev("autoplay_error", qid=qid, error="no such question in qa20")
            self.finish(ok=False)
            return
        ex = self.rect(self.entry)
        self.move_pointer(ex[0] + self.s(60), (ex[1] + ex[3]) // 2, 0.45,
                          lambda: self.type_text(qid, text))

    def type_text(self, qid, text):
        self.entry.focus_force()
        self.entry.delete(0, "end")
        self.log.ev("type_start", qid=qid)

        def step(k=0):
            if k < len(text):
                self.entry.insert("end", text[k])
                self.entry.xview_moveto(1.0)
                self.tk.after(self.a.type_ms, step, k + 1)
            else:
                self.log.ev("type_end", qid=qid)
                bx = self.rect(self.button)
                self.tk.after(200, lambda: self.move_pointer((bx[0] + bx[2]) // 2, (bx[1] + bx[3]) // 2, 0.4,
                                                             lambda: self.press(qid)))
        step()

    def press(self, qid):
        self.button.config(bg=COLORS["accent_dn"])
        self.log.ev("press", qid=qid)

        def release():
            self.button.config(bg=COLORS["accent"])
            self.ask(qid)
        self.tk.after(140, release)

    def move_pointer(self, x1, y1, seconds, done):
        if os.name != "nt":
            done()
            return
        import ctypes.wintypes as wt
        p = wt.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(p))
        x0, y0 = p.x, p.y
        t0 = time.perf_counter()
        self.log.ev("pointer", frm=[x0, y0], to=[int(x1), int(y1)], seconds=seconds)

        def step():
            u = min(1.0, (time.perf_counter() - t0) / seconds)
            e = u * u * (3 - 2 * u)
            ctypes.windll.user32.SetCursorPos(int(round(x0 + (x1 - x0) * e)), int(round(y0 + (y1 - y0) * e)))
            if u < 1.0:
                self.tk.after(16, step)
            else:
                done()
        step()

    def finish(self, ok=True):
        turns = getattr(self, "done_turns", [])
        fin = {"ok": bool(ok) and len(turns) == len(self.autoplay) and not any(self.failed),
               "turns": [{"qid": r["qid"], "text": r["text"], "answers": r["answers"]} for r in turns],
               "failed": self.failed, "t": time.time()}
        self.log.write("finished.json", fin)
        self.log.ev("finished", ok=fin["ok"])
        if self.a.quit_file:
            self.wait_quit()
        elif self.a.exit_after:
            self.quit()

    def wait_quit(self):
        if os.path.exists(self.a.quit_file):
            self.quit()
        else:
            self.tk.after(100, self.wait_quit)

    def quit(self):
        self.finished = True
        for w in self.workers:
            if w:
                w.stop()
        self.log.ev("quit")
        self.tk.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=os.path.join(HERE, "scene_npu.json"))
    ap.add_argument("--root", default="")
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--autoplay", default="", help="qa20 ids to type and ask after the go file, e.g. q05,q06")
    ap.add_argument("--go-file", default="")
    ap.add_argument("--quit-file", default="")
    ap.add_argument("--exit-after", action="store_true", help="quit when the autoplay ends (no quit file)")
    ap.add_argument("--type-ms", type=int, default=35)
    ap.add_argument("--lead-ms", type=int, default=500)
    ap.add_argument("--hold-ms", type=int, default=1600)
    ap.add_argument("--end-ms", type=int, default=400)
    a = ap.parse_args()
    App(a).tk.mainloop()


if __name__ == "__main__":
    main()
