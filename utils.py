import os, json
import torch
import subprocess

def save_json(fname, data):
    create_dirs(fname)
    with open(fname, "w") as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True)

    print("Saved %s" % fname)

def run_bash_command(command, noSplit=True):
    if noSplit:
        command = command.split()
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    output, error = process.communicate()

    return str(output)

def load_json(fname):
    with open(fname, "r") as json_file:
        d = json.load(json_file)
    
    return d
def extract_fname(directory):
    import ntpath
    return ntpath.basename(directory)

def create_dirs(fname):
    if "/" not in fname:
        return
        
    if not os.path.exists(os.path.dirname(fname)):
        try:
            os.makedirs(os.path.dirname(fname))
        except OSError:
            pass     

def imsave(fname, img):
    import pylab as plt
    plt.imsave(fname,img,cmap="gray")

# @torch.enable_grad()
def compute_grad(loss_function, x, Z):
    x_tmp = torch.FloatTensor(x.clone().data)
    x_tmp.requires_grad=True

    if x_tmp.grad is not None:
        x_tmp.grad.zero_()

    loss_function(x_tmp, Z).backward()

    G = x_tmp.grad.detach()

    return G 

def load_history(d, m, l, e, s, r):
    exp_name = "{}-{}-{}-{}-{}".format(d,m,l,e,s)
    #path_save = "/mnt/home/issam/manSAGA/Saves/{}.json".format(exp_name)
    #path_model = "/mnt/home/issam/manSAGA/Saves/{}.pth".format(exp_name)
    path_save = "checkpoints/{}.json".format(exp_name)
    path_model = "checkpoints/{}.pth".format(exp_name)

    if os.path.exists(path_save) and not r:
        return load_json(path_save)

    else:
        return {"exp_name_no_lr":"{}-{}-{}-{}".format(d,m,e,s),
                "exp_name":exp_name,
                "path_model":path_model,
                "path_save":path_save,
                "path_plot":"figures/{}.png".format(exp_name),
                "loss":[]}

    return exp_name

def save_image(x, shape, i, prefix):
    pathies = "/mnt/home/issam/Summaries/manSAGA/qualitative/%s/"% prefix.replace(".json","")
    create_dirs(pathies + "tmp")
    img_name = "%s/%f.png" % (pathies, i)

    imsave(img_name, x.reshape(shape))
    print(img_name)


n2d = {"synthetic": "Synthetic",
                    "Mnist":"MNIST",
                    "ocean":"Ocean"}


l2l = lambda x: "%s %s" % (x.split("_")[0], x.split(" ")[1])

# def get_color_label(m,s):
#     if m == "" and s == "":
#         pass
#     colors = ['#741111', "#000000", '#3a49ba','#7634c9', 
#               "#4C9950", "#CC29A3", '#ba3a3a', "#0f7265",
#               "#7A7841", "#00C5CD", "#6e26d9"]

# def get_marker(m,s):
#     if m == "" and s == "":
#         pass
#     ls_markers = [("-", "o"), ("-", "p"), ("-", "D"), ("-", "^"), ("-", "s"),
#                ("-", "8"), ("-", "o"), ("-", "o"), ("-", "o"), ("-", "o"), 
#                ("-", "o"), ("-", "o")]
def get_plot_label(m, s, l):
    s2s = {"uniform":"U", "lipschitz":"NU"}
    m2m = {"svrg":"RSVRG", "sgd":"RSGD", "saga":"MASAGA"}
    return l2l("%s_$10^{-%s}$ (%s)" % (m2m[m],str(l)[-1], s2s[s]))                            
# def get_best():
#     scores = []
#     for d, m, e, l, s in product(dList, mList, eList, lList, sList):
#             fname = ut.get_exp_path(d, m, l, e, s)
#             if not os.path.exists(fname):
#                 print("Skipped in %s" % fname)
#             else:
#                 history = pd.DataFrame(ut.load_json(fname)["loss"])
#                 scores += [{"d":d,
#                             "method":"%s_1e%d (%s)" % (m, int(round(np.log(l)/np.log(10))), s2s[s]),
#                             "loss":np.array(history["loss"])[-1], 
#                             "epoch":np.array(history["epoch"])[-1],
#                             "fname":fname}]
#     if len(scores) > 0:
#         print()
#         df = pd.DataFrame(scores)

#         best_scores = []
#         for i in ["saga", "svrg", "sgd"]:
#             for j in ["\(U\)", "\(NU\)"]:

#                 rr = df[(df['method'].str.contains(i)) & (df['method'].str.contains(j))].sort_values(
#                     by=['loss'],ascending=True)

#                 if len(rr) == 0:
#                     print("Skipped in (%s,%s)" % (i,j))
#                     continue

#                 best_scores += [dict(rr.iloc[0])]

#         #print(df.sort_values(by=['loss'],ascending=False))

#         return best_scores