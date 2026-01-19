def analysis():
    networks = ["icdt", "colt", "pods", "pkdd", "ecml", "sdm", "stacs", "uai", "edbt", "stoc", "soda",
                "focs", "icml", "icdm", "vldb", "www", "kdd", "sigmod", "icde", "ai", "th", "db", "dm", "dblp"]
    ts = [i for i in range(4, 21)]
    j = 0
    for network in networks:
        print(network)
        for algo in ["cmopso"]:
            open("/home/ramesh/dblp/output/" + network + "_" + algo + "_analysis.txt", "w").close()
            try:
                with open("/home/ramesh/dblp/output/" + network + "_" + algo + "_results.txt", "r") as file:
                    lc = 0
                    fsum = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                    for line in file:
                        lc += 1
                        result = [x for x in line.strip("\n").split("\t") if x]
                        tsum = [fsum[i] + float(result[i]) for i in range(len(result))]
                        fsum = tsum
                        if lc % 5 == 0:
                            j += 1
                            open("/home/ramesh/dblp/output/" + network + "_" + algo + "_analysis.txt", "a").write(
                                "\t".join([str(fsum[i] / 5) for i in range(len(result))]) + "\n")
                            fsum = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                            tsum.clear()
            except Exception as e:
                print("The error is: ",e)

if __name__ == '__main__':
    analysis()
