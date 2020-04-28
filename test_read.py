split = "train"
file_path = split + ".txt"
root = "Caltech101"
i = 0
with open(file_path, "r" ) as fp:
    for line in fp:
        row = line.rstrip("\r\n")
        print(root + "/" + row + " " + str(i))
        # item = data_elem(img,row.split("/")[0])
        # my_dataset.append(item)
        i += 1
