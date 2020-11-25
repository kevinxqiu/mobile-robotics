from threading import Event

exit = Event()

def do_my_thing():
    print(test)
def main():
    while not exit.is_set():
      do_my_thing()
      exit.wait(60)

    print("All done!")
    # perform any cleanup here

def quit(signo, _frame):
    print("Interrupted by %d, shutting down" % signo)
    exit.set()

if __name__ == '__main__':

    import signal
    for sig in ('TERM', 'HUP', 'INT'):
        signal.signal(getattr(signal, 'SIG'+sig), quit);

    main()