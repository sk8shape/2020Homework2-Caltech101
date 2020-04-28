with open ("test.txt") as f:
    row = f.read()
    string = row.rstrip("\r\n")
    print (string)
