import struct


# def pad_one(format):
#     p = 0
#     padded_format = ''
#     align = 1
#     for s in format:
#         size = struct.calcsize(s)
#         align = max(align, size)
#         while p % size:
#             padded_format += 'x'
#             p += 1
#         padded_format += s
#         p += size
#     while p % align:
#         padded_format += 'x'
#         p += 1
#     return padded_format, p, align
#
#
# def pad(format_list):
#     p = 0
#     padded_format = ''
#     align = 1
#     for format in format_list:
#         padded_sublayout, subsize, subalign = pad_one(format)
#         while p % subalign:
#             padded_format += 'x'
#             p += 1
#         padded_format += padded_sublayout
#         p += subsize
#         align = max(align, subalign)
#     while p % align:
#         padded_format += 'x'
#         p += 1
#     return padded_format, p, align


def pad(abstract_layout, depth=0):
    i = 0
    padded_layout = ''
    padded_size = 0
    alignment = 1
    while i < len(abstract_layout):
        print('%s%s  %s' % ('  ' * depth, abstract_layout[i:], padded_layout))
        c = abstract_layout[i]
        if c == '[':
            step, padded_sublayout, padded_subsize, subalign = pad(abstract_layout[i + 1:], depth + 1)
            alignment = max(alignment, subalign)
            print('SUB', step, padded_sublayout, padded_subsize, subalign)

            while padded_size % subalign:
                padded_layout += 'x'
                padded_size += 1
            padded_layout += padded_sublayout
            padded_size += padded_subsize

            i += step + 1
        elif c == ']':
            # return i + 1
            return i + 1, padded_layout, padded_size, alignment
        else:
            c_size = struct.calcsize(c)
            alignment = max(alignment, c_size)
            while padded_size % c_size:
                padded_layout += 'x'
                padded_size += 1
            padded_layout += c
            padded_size += c_size

            i += 1
    while padded_size % alignment:
        padded_layout += 'x'
        padded_size += 1
    print('Final align', alignment)
    print('Final size', padded_size)
    print('Final spec', padded_layout)


if __name__ == '__main__':

    # for format in ['f', 'fc', 'cf', 'chf', 'hcf', 'hcfd']:
    #     print(struct.calcsize(pad_one(format)[0]), struct.calcsize(format))

    # for format in [['df', 'fc']]:
    #     f, p, a = pad(format)
    #     print(f, p, a, struct.calcsize(f))

    # pad_recursive('[[f][cc]]')
    pad('[[ch][fdh]]')
