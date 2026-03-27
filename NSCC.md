# Set up NSCC account and SSH

NUS students get [100,000 SU](https://nsccsg.github.io/policies/resource-allocation-policy/#resource-allocation-to-personal-projects-aspire-2a-only)

1. Make sure you are on NUS VPN
2. Login via [User Portal](https://user.nscc.sg)
   1. Reset your SSH key, copy and save it into your `~/.ssh` directory
      ```bash
      nano ~/.ssh/id_rsa_nscc
      # paste SSH key -> ctrl-x -> y -> enter
      ```
   2. Add `nscc` to your SSH config (`~/.ssh/config`) (replace `<username>`)
      ```
      Host nscc
        HostName aspire2a.nus.edu.sg
        User <username>
        IdentityFile ~/.ssh/id_rsa_nscc
      ```
   3. set/reset NSCC password
3. Test your ssh setup
   ```bash
   ssh nscc
   ```
4. You can also view the [Web Portal](https://nusjobportal.nscc.sg)

# Set up `uv` on NSCC

1. ssh into NSCC
2. run install script
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. set UV cache dir
   ```bash
   mkdir -p ~/scratch/.cache/uv
   echo "export UV_CACHE_DIR=~/scratch/.cache/uv" >> ~/.bashrc
   ```

# Useful commands

## Temporary interactive session

```bash
qsub -I -q normal -l select=1:ngpus=1 -l walltime=00:10:00 -P personal
```

## Check GPU usage

```bash
qstat # get job id (e.g. `13405122.pbs101`)
qstat -n 13405122 # get last output e.g. x1000c0s7b0n0/2*16
export PBS_JOBID=13405122.pbs101
ssh x1000c1s3b0n0 # ssh into the node
watch -n 1 nvidia-smi
```
