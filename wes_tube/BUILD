python_sources(
    name="lib",
    dependencies=[
        "//:3rdparty#sqlalchemy",
        "//:3rdparty#alembic",
        "//:3rdparty#pyav",
        "//:3rdparty#requests",
        "//:3rdparty#pyside6",
    ],
)

pyoxidizer_binary(
    name="westube",
    entry_point="wes_tube.cli.main:main",
    dependencies=[
        ":lib",
    ],
)
