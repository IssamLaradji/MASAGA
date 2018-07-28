import utils as ut

borgy_script = r'''
borgy submit --req-gpus=0 --req-cores=1 --req-ram-gbytes=12 \
            -v /mnt/home/issam:/mnt/home/issam -v /mnt/datasets:/mnt/datasets\
            -i images.borgy.elementai.lan/issam/v1 \
            -w /home/issam/Research_Ground/manSAGA/ \
            --name issam -- /bin/bash -c command
'''



#borgy_script = borgy_script.replace("End2End", PROJECT_NAME)
def run(dataset_name, model_name,
        epochs, learning_rate, sampling_method):
  argDict = {"m":model_name, "d":dataset_name, "e":epochs, "l":learning_rate, "s":sampling_method}
  command = get_command(argDict)
  return borgy_submit(command, force=True)

def get_command(d,m,e,l,s,r):
  command = (("python main.py -e tmp -dList {} -mList {}"+
             " -eList {} -lList {} -sList {} -r {}").format(d,m,e,l,s,r))

  return command 



def is_same(cmd, b_cmd):
  b_cmd = b_cmd[b_cmd.find("-c")+3:]

  flag = True

  for el in [a for a in cmd.split() if "-"==a[0]]:
    if extract(cmd, el) != extract(b_cmd, el):
      return False


  return flag

def run_command(command, force=False):
  jobid, job_state = get_job_id(command)
  import ipdb; ipdb.set_trace()  # breakpoint 7153b5bf //

  if job_is_running(job_state):
    return "Already Running"

  cmm = borgy_script.replace("command", 
                       '"%s"' % command)

  if force is True:
    #print("STARTING: %s" % command)
    subprocess.call([cmm], shell=True)

    return "Launching"

  else:
    prompt = input("Do you want to borgy submit the command:"
                   " \n'%s' ? \n(y/n)\n" % command) 
    if prompt == "y":
      subprocess.call([cmm], shell=True)
      return "Launching"

  #print("SKIPPED: %s" % command)
  return "Skipped"

def get_job_id(command):
    cmdList = ut.run_bash_command("borgy ps | grep RUNNING").split("\\n")
    
    jobid = None
    status = None
    jobid_failed = None
    status_failed = None

    for cmd in cmdList:
      try:
        tmp_jobid, tmp_status = cmd.split()[:2]
      except:
        continue 

      if tmp_status in ["RUNNING", "QUEUED", "QUEUING", "FAILED"]:
        if is_same(command, cmd):

          if tmp_status == "FAILED":
            jobid_failed = tmp_jobid
            status_failed = tmp_status

          else:
            jobid = tmp_jobid
            status = tmp_status

    if jobid is None:
      jobid = jobid_failed
      status = status_failed

    return jobid, status




def job_is_running(job_state):
  return job_state in  ["RUNNING", "QUEUED", "QUEUING"]


import subprocess
def borgy_display(jobid):
    subprocess.call(["borgy logs %s" % jobid], shell=True)




def borgy_kill(jobid, force=False):
    if jobid is None:
        return "Not Running"
        
    if force is True:
      subprocess.call(["borgy kill %s" % jobid], shell=True)
      return "Soon"

    else:
      prompt = input("Do you want to delete : \n'%s'"
                     " ? \n(y/n)\n" % (str(jobid))) 
      if prompt == "y":
          subprocess.call(["borgy kill %s" % jobid], shell=True)
          return "Deleted"
      else:
          print("Skipped")
          return "Skipped"
     

def extract(cmd, which="-p"):
    if cmd[0] == "-":
      which = which + " "
    else:
      which = " " + which + " "
    
    sindex = cmd.rfind(which)

    if sindex == -1:
      return None

    sb = cmd[sindex+len(which):]
    next_space = sb.find(" ")

    if next_space == -1:
        return sb.strip()
    else:
        return sb[:next_space].strip()




