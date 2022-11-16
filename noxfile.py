import nox


@nox.session
def tests(session):
    session.install(".[dev]")
    session.run("pytest")
