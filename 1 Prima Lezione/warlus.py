measure = [2, 8, 0, 1, 1, 9, 7, 7]

info_measure = {
    "length": len(measure),
    "sum":    sum(measure),
    "mean":   sum(measure) / len(measure),
}

print(info_measure)


info_measure = {
    "length": (length := len(measure)),
    "sum":    (total  := sum(measure)),
    "mean":   total / length,
}

print(info_measure)
print(length)
print(total)