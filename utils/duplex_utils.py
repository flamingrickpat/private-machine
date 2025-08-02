import sys, os, time, queue

def duplex_com(queue_in, queue_out, my_name, other_name):
    """
    Full-duplex console chat.
    - queue_in:  Queue of incoming messages (from the other side).
    - queue_out: Queue to send your messages to the other side.
    - my_name / other_name: Labels shown when printing messages.

    Controls:
      - Type to compose. Press Enter to send the current line.
      - Backspace edits the line.
      - Ctrl-C (or Ctrl-D on Unix) exits.

    Notes:
      - Incoming messages are printed immediately, even while you're typing.
      - On each interrupt (incoming or outgoing), the speaker name is shown.
    """
    is_windows = os.name == "nt"

    # -------- helpers (prompt rendering, safe prints) --------
    prompt_prefix = f"{my_name}> "
    buf = []

    def buf_text():
        return "".join(buf)

    def clear_line_and_print(s):
        # \r to line start, \x1b[K to clear line
        sys.stdout.write("\r\x1b[K" + s)
        sys.stdout.flush()

    def render_prompt():
        clear_line_and_print(prompt_prefix + buf_text())

    def println(s=""):
        sys.stdout.write(s + "\n")
        sys.stdout.flush()

    # -------- platform-specific key input --------
    if is_windows:
        import msvcrt

        def read_key_nonblocking():
            if msvcrt.kbhit():
                ch = msvcrt.getwch()  # wide char
                return ch
            return None

        def is_enter(ch): return ch in ("\r", "\n")
        def is_backspace(ch): return ch in ("\b", "\x7f")
        def is_ctrl_c(ch): return ch == "\x03"
        def is_ctrl_d(ch): return False  # not used on Windows

    else:
        import tty, termios, select
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        def read_key_nonblocking():
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if r:
                return sys.stdin.read(1)
            return None

        def is_enter(ch): return ch in ("\r", "\n")
        def is_backspace(ch): return ch in ("\x7f", "\b")
        def is_ctrl_c(ch): return ch == "\x03"
        def is_ctrl_d(ch): return ch == "\x04"

        # put terminal into cbreak (near-raw) mode
        tty.setcbreak(fd)

    # -------- main loop --------
    try:
        println(f"Started. Press Enter to send; Ctrl-C to quit{' (Ctrl-D also on Unix)' if not is_windows else ''}.")
        render_prompt()

        while True:
            # 1) Drain any incoming messages (interrupts)
            interrupted = False
            while True:
                try:
                    msg = queue_in.get_nowait()
                except queue.Empty:
                    break
                else:
                    # Incoming interrupt: show other_name and message
                    clear_line_and_print(f"{other_name}: {msg}")
                    println()
                    interrupted = True

            if interrupted:
                # Repaint your prompt and current buffer after showing interrupts
                render_prompt()

            # 2) Handle local keystrokes (non-blocking)
            ch = read_key_nonblocking()
            if ch is not None:
                if is_ctrl_c(ch) or is_ctrl_d(ch):
                    # Show who interrupted (you) when exiting mid-typing
                    if buf:
                        clear_line_and_print(f"{my_name} (interrupt): {buf_text()}")
                        println()
                    println("Exiting.")
                    break

                if is_enter(ch):
                    text = buf_text()
                    # Outgoing interrupt: you speak now
                    clear_line_and_print(f"{my_name}: {text}")
                    println()
                    if text.strip():
                        queue_out.put(text)
                    buf.clear()
                    render_prompt()
                elif is_backspace(ch):
                    if buf:
                        buf.pop()
                        render_prompt()
                else:
                    # Printable char(s)
                    # Filter out non-printing control characters
                    if ch.isprintable() or ch == "\t":
                        buf.append(ch)
                        # update line with new buffer
                        render_prompt()

            # 3) Small idle to reduce CPU
            time.sleep(0.01)

    finally:
        # restore terminal on Unix
        if not is_windows:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        println()  # ensure cursor on next line


# ---------- Optional demo ----------
if __name__ == "__main__":
    import threading, queue, time

    q_in = queue.Queue()   # messages coming from "them" to us
    q_out = queue.Queue()  # messages we send out

    # A tiny echo bot to demonstrate full-duplex interrupts
    def bot_worker():
        while True:
            try:
                msg = q_out.get(timeout=0.1)
            except queue.Empty:
                continue
            if msg.lower() in {"quit", "exit"}:
                break
            # Simulate thinking time and mid-typing interruptions
            for chunk in ["Got it,", " working", " on it..."]:
                time.sleep(0.4)
                q_in.put(chunk)
            time.sleep(0.6)
            q_in.put(f"Echo: {msg}")

    t = threading.Thread(target=bot_worker, daemon=True)
    t.start()

    duplex_com(q_in, q_out, my_name="You", other_name="Bot")
