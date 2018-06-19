# ephys_analysis
Analysis routines for ephys fhd outputs

## Installation
### Using scipy-notebook docker
Run the follwoing command to run the docker:
```
sudo docker run -it -p 8890:8888 --user root -e GRANT_SUDO=yes -e NB_USER=kkarbasi -e NB_UID=1001 -e NB_GID=1001 -e CHOWN_HOME=yes -e CHOWN_HOME_OPTS='-R' -v /mnt/data/:/mnt/data -v /home/kkarbasi/docker_mount:/home/kkarbasi/run:Z --name ephys_analysis docker.io/jupyter/scipy-notebook
```

