"""
Code to manage hiding and restoring files in a Git repository
"""
import argparse
import subprocess
from pathlib import Path

GITIGNORE = Path(".gitignore")

def run(cmd):
    '''
    Execute a shell command using subprocess.
    The command is printed to stdout before execution.
    If the command exits with a non-zero status, an exception is raised.

    Parameters
    ----------
    cmd : list[str]
        Command and arguments to execute (e.g. ["git", "status"]).
    '''
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)

def toggle(path, enable):
    '''
    Enable or disable a path entry inside the .gitignore file.
    If enable is True,  the path is written as an active ignore rule.
    If enable is False, the path is commented out.

    Parameters
    ----------
    path : str
        Path (file or directory) to toggle inside .gitignore.
    enable : bool
        Whether the path should be enabled (True) or commented out (False).

    Raises
    ------
    ValueError
        If the given path is not present in .gitignore.
    '''
   
    lines = GITIGNORE.read_text().splitlines()
    out   = []
    found = False

    for line in lines:
        stripped = line.lstrip("#").strip()
        
        if stripped == path:
            found = True
            if enable:
                out.append(path)
            else:
                out.append(f"# {path}")
        else:
            out.append(line)

    if not found:
        raise ValueError(f"Path {path} not found in .gitignore")
    
    GITIGNORE.write_text("\n".join(out) + "\n")


def hide(path):
    '''
    Hide a file or directory from Git tracking.

    This function:
    1. Enables the path in .gitignore.
    2. Removes the path from the Git index (without deleting it locally).
    3. Commits the change.
    4. Pushes the commit to the remote repository.

    Parameters
    ----------
    path : str
        File or directory to hide.
    '''
    toggle(str(path), 1)
    run(["git", "rm", "--cached", "-r", str(path)])
    run(["git", "add", ".gitignore"])
    run(["git", "commit", "-m", f"Hide {path}"])
    run(["git", "push"])


def restore(path):
    '''
    Restore a previously hidden file or directory to Git tracking.

    This function:
    1. Disables the path in .gitignore (comments it out).
    2. Re-adds the path to the Git index.
    3. Commits the change.
    4. Pushes the commit to the remote repository.

    Parameters
    ----------
    path : str
        File or directory to restore.
    '''
    toggle(str(path), 0)
    run(["git", "add", ".gitignore"])
    run(["git", "add", str(path)])
    run(["git", "commit", "-m", f"Restore {path}"])
    run(["git", "push"])


def main():
    '''
    Entry point of the script.
    Parses command-line arguments and dispatches the requested
    action (hide or restore) for one or more paths.

    Supported commands:
    - hide <paths...>
    - restore <paths...>
    '''
    parser = argparse.ArgumentParser(
        description="Hide / restore files for course repositories"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    hide_p = sub.add_parser("hide", help="Ignore path(s) and remove them from tracking")
    hide_p.add_argument(
        "paths",
        nargs="+",
        help="One or more files or directories to hide"
    )

    res_p = sub.add_parser("restore", help="Restore previously hidden path(s)")
    res_p.add_argument(
        "paths",
        nargs="+",
        help="One or more files or directories to restore"
    )

    args = parser.parse_args()

    try:
        if args.command == "hide":
            for p in args.paths:
                hide(p)
        elif args.command == "restore":
            for p in args.paths:
                restore(p)
    except Exception as e:
        print(f"{e}")
        return


if __name__ == "__main__":
    main()
