#! /usr/bin/env python
import subprocess

def main():
    errno = subprocess.call(['py.test'])
    raise SystemExit(errno)

if __name__ == '__main__':
    main()
