from onnx_vectorize import vectorize

if __name__ == '__main__':

    # Enter your own file path
    file_path = '<file path>'

    d, l, p = vectorize(file_path)

    print('Dimension Vector')
    print(d)
    print('Layer Vector')
    print(l)
    print('Parameter Vector')
    print(p)