def is_luhn_valid(number: str) -> bool:
    """
    Check if the given string passes the Luhn algorithm.
    """
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13:
        return False

    checksum = 0
    reverse = digits[::-1]

    for i, d in enumerate(reverse):
        if i % 2 == 1:
            doubled = d * 2
            if doubled > 9:
                doubled -= 9
            checksum += doubled
        else:
            checksum += d

    return checksum % 10 == 0
