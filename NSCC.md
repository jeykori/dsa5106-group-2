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
