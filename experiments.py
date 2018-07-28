def get_experiment(experiment, args):
        if experiment == "all":
            dList = ["M04", "M59", "A", "B"]
            mList = ["sgd", "saga"]
            eList = [100]
            lList = [1e1, 1.0, 1e-1,1e-2,1e-3,1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
            sList = ["uniform",  "lipschitz"]

        if experiment == "synthetic":
            dList = ["synthetic"]
            mList = ["svrg", "saga", "sgd"]
            eList = [100]
            lList = [1e-1,1e-2,1e-3,1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
            sList = ["lipschitz", "uniform"]

        if experiment == "mnist":
            dList = ["Mnist"]
            mList = ["svrg", "saga", "sgd"]
            eList = [10]
            lList = [1e-1,1e-2,1e-3,1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
            sList = ["lipschitz", "uniform"]

        if experiment == "mnist_L":
            dList = ["Mnist"]
            mList = ["svrg", "saga", "sgd"]
            eList = [20]
            lList = ["L"]
            sList = ["lipschitz", "uniform"]

        if experiment == "ocean_L":
            dList = ["ocean"]
            mList = ["svrg", "saga", "sgd"]
            eList = [50]
            lList = ["L"]
            sList = ["lipschitz", "uniform"]

        if experiment == "synthetic_L":
            dList = ["synthetic"]
            mList = ["saga", "svrg", "sgd"]
            eList = [100]
            lList = ["L"]
            sList = ["uniform","lipschitz"]

        if experiment == "mnistAll":
            dList = ["M1", "M2","M3", "M4", "M5", "M6"]
            mList = ["saga"]
            eList = [10]
            lList = [1e-5]
            sList = ["uniform"]

        if experiment == "ocean":
            dList = ["ocean"]
            mList = ["svrg", "saga", "sgd"]
            eList = [50]
            lList = [1e-1,1e-2,1e-3,1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
            sList = ["uniform","lipschitz"]

        if experiment == "T":
            dList = ["A"]
            mList = ["svrg"]
            eList = [500]
            lList = [1e-8]
            sList = ["uniform"]


        if experiment == "B":
            dList = ["B"]
            mList = ["sgd", "saga"]
            eList = [100]
            lList = [1e1, 1.0, 1e-1,1e-2,1e-3,1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
            sList = ["uniform",  "lipschitz"]

        if experiment == "M04":
            dList = ["M04"]
            mList = ["sgd", "saga"]
            eList = [100]
            lList = [1e1, 1.0, 1e-1,1e-2,1e-3,1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
            sList = ["uniform","lipschitz"]

        if experiment == "M59":
            dList = ["M59"]
            mList = ["sgd", "saga"]
            eList = [100]
            lList = [1e1, 1.0, 1e-1,1e-2,1e-3,1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
            sList = ["uniform","lipschitz"]

        dList = args.dList or dList
        mList = args.mList or mList
        eList = args.eList or eList
        lList = args.lList or lList
        sList = args.sList or sList


        return dList, mList, eList, lList, sList