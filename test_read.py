i = 0
with open ("test.txt") as f:

    for line in f:
        print(line.split("/")[0])
        print (line.rstrip("\n\r") + " " + str(i))
        i += 1
