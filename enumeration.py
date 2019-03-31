def enumeration(length):
    enum_list = []
    fds([], length, enum_list)
    return enum_list

def fds(sub_list, length, enum_list):
    if len(sub_list) == length:
        enum_list.append(sub_list)
        return
    for i in [-1, 1]:
        fds(sub_list + [i], length, enum_list)

a = enumeration(3)
