"""Scip status codes."""

status = {
    "unknown": 0,
    "userinterrupt": 11,
    "nodelimit": 8,
    "totalnodelimit": 8,
    "stallnodelimit": 8,
    "timelimit": 9,
    "memlimit": 0,
    "gaplimit": 13,
    "sollimit": 10,
    "bestsollimit": 10,
    "restartlimi": 0,
    "optimal": 2,
    "infeasible": 3,
    "unbounded": 5,
    "inforunbd": 4,
    "terminate": 11
}
