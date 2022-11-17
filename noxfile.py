import nox


@nox.session
def tests(session):
    session.install(".[test]")
    session.run("pytest")
