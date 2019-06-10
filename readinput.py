# import the python subprocess module
import subprocess
 
 
def run_cmd(args_list):
        """
        run linux commands
        """
        # import subprocess
        print('Running system command: {0}'.format(' '.join(args_list)))
        proc = subprocess.Popen(args_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        s_output, s_err = proc.communicate()
        s_return =  proc.returncode
        return s_return, s_output, s_err 
        

(ret, out, err)= run_cmd(['hdfs', 'dfs', '-get', '/user', 'home/sampath/Desktop/project'])
#lines = out.split('\n')

hdfs_file_path = '/user/sampu'
cmd = ['hdfs', 'dfs', '-test', '-d', hdfs_file_path]
ret, out, err = run_cmd(cmd)
print(ret, out, err)

