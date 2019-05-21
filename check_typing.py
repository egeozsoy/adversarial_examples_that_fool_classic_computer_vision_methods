from mypy.api import run


if __name__ == '__main__':
    result = run(['src.py','configurations.py','helpers','custom_configurations'])
    if result[0]:
        print('\nType checking report:\n')
        print(result[0])  # stdout

    if result[1]:
        print('\nError report:\n')
        print(result[1])  # stderr

    print('\nExit status:', result[2])