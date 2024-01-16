from options import options_predict

def main():
    # unpack values in parser
    args = options_predict()
    print(args)

if "__main__" == __name__:
    main()
