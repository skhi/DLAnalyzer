#!/usr/bin/env bash

# Author: Ryan TIffany
# Email: rtiffany@us.ibm.com

# Script Variables
DIALOG='\033[0;36m' 
WARNING='\033[0;31m'
LINKY='\033[0;41m'
NC='\033[0m'
today=$(date "+%F")
hst=$(hostname -s)

# Short description 
overview() { 
  echo -e "${DIALOG}This script will install s3cmd and rsnapshot on your server and help with some basic backup configurations.${NC}"
  echo -e "${DIALOG}By default rsnapshot is configured to only backup this system, but can be configured to backup remote systems as well.${NC}\n"
}

overview 

check_your_privilege () {
    SUDO=''
    if [[ "$(id -u)" != 0 ]]; then
        SUDO='sudo'
        echo -e "${WARNING}Note: This setup script requires root permissions. You will be prompted for your sudo password.${NC}"
    fi
}

check_your_privilege

set_install_variables()	{

  # OS specific variables.
  if [ -e /etc/redhat-release ] ; then
    echo -e "${DIALOG}Red Hat based distribution detected...${NC}"
    OS_VENDOR=$(awk '{print $1}' /etc/redhat-release | tr '[:lower:]' '[:upper:]')
    if [ "${OS_VENDOR}" = "RED" ]; then
          OS_VENDOR="REDHAT"
    elif [ "${OS_VENDOR}" = "CENTOS" ]; then
          OS_VENDOR="CENTOS"
    fi
    OS_INSTALL_TOOL="/usr/bin/yum -y install"
    MAJOR_VERSION=$(rpm -qa \*-release | grep -Ei "oracle|redhat|centos" | cut -d"-" -f3)
      if [ "${OS_VENDOR}" = "REDHAT" ] && [ "${MAJOR_VERSION}" = "6" ]; then
        $SUDO rpm -Uvh http://download.fedoraproject.org/pub/epel/6/x86_64/epel-release-6-8.noarch.rpm >/dev/null
      elif [ "${OS_VENDOR}" = "CENTOS" ] && [ "${MAJOR_VERSION}" = "6" ]; then
        $SUDO rpm -Uvh http://download.fedoraproject.org/pub/epel/6/x86_64/epel-release-6-8.noarch.rpm >/dev/null 
      elif [ "${OS_VENDOR}" = "REDHAT" ] && [ "${MAJOR_VERSION}" = "7" ]; then
        $SUDO yum install epel-release -y >/dev/null 
      elif [ "${OS_VENDOR}" = "CENTOS" ] && [ "${MAJOR_VERSION}" = "7" ]; then
        $SUDO yum install epel-release -y >/dev/null 
      fi
  elif [ -e /usr/bin/lsb_release ] ; then
    echo -e "${DIALOG}Debian based distribution detected...${NC}"
    OS_VENDOR=$(lsb_release -si | tr '[:lower:]' '[:upper:]')
    #OS_VERSION_MAJOR=$(lsb_release -sr | cut -d. -f1)
    #OS_VERSION_MINOR=$(lsb_release -sr | cut -d. -f2)
    OS_INSTALL_TOOL="apt-get -y install"
  fi
}

install_tools() {
	$SUDO $OS_INSTALL_TOOL s3cmd rsync rsnapshot wget >/dev/null 
}

configure_rsnapshot()
{
  sudo mv /etc/rsnapshot.conf{,.bak}
  sudo wget -O /etc/rsnapshot.conf https://gist.githubusercontent.com/greyhoundforty/6b6975d973f5550fce69b71ed8485d34/raw/4091bb17f03dc9b0a3f745b348e686db14b4027e/rsnapshotv2.conf
  echo
  echo
  echo -n -e "${DIALOG}Please supply the directory you would like to use to store your backups. Use the full path with a trailing slash (example: /backups/)${NC}  "
  read -r RSNAPSHOT_BACKUP_DIR
  echo -e "\n${DIALOG}Set rsnapshot backup directory to ${RSNAPSHOT_BACKUP_DIR} ${NC}  "
  
  sudo sed -i "s|BACKUP_DIR|$RSNAPSHOT_BACKUP_DIR|" /etc/rsnapshot.conf
  echo -e "${DIALOG}Testing rsnapshot configuration.${NC}\n"
  rsnapshot configtest
}

configure_s3cmd() { 

	wget -O "$HOME/.s3cfg" https://gist.githubusercontent.com/greyhoundforty/676814921b8f4367fba7604e622d10f3/raw/f6ce1f2248c415cefac4eec4f1c112ad4a03a0d1/s3cfg
	echo 
	echo 
	echo -n -e "${DIALOG}Please supply your Cloud Object Storage (S3) Access Key:${NC}  "
	read -r -s COS_ACCESS_KEY
	sed -i "s|cos_access_key|$COS_ACCESS_KEY|" "$HOME"/.s3cfg
  echo ""
	echo -n -e "${DIALOG}Please supply your Cloud Object Storage (S3) Secret Key:${NC}  "
	read -r -s COS_SECRET_KEY
	sed -i "s|cos_secret_key|$COS_SECRET_KEY|" "$HOME/.s3cfg"
	echo 
	echo -n -e "${DIALOG}Please supply your Cloud Object Storage (S3) Endpoint:${NC}  "
	read -r ENDPOINT 
  sed -i "s|cos_endpoint|$ENDPOINT|g" "$HOME/.s3cfg"
  echo ""
  echo -e "${DIALOG}We will now test our config file by creating a test bucket based on your systems hostname.${NC}"
  $(which s3cmd) mb s3://"$hst"
  if [ $(s3cmd ls | egrep "$hst" | wc -l) = "1" ];then
    echo -e "${DIALOG}s3cmd configuration test passed. Now Removing test bucket.${NC}"
  $(which s3cmd) rb s3://"$hst" 
  else
    echo -e "${WARNING}Error: Bucket creation did not succeed, double check your HOME/.s3cfg configuration file.${NC}"
  fi
  
}

# Set a basic daily cron to compress our rsnapshot backup directory and send it to s3cmd 
cos_backup_schedule() { 
echo -e "${DIALOG}Setting Daily cron to send backups to Cloud Object Storage${NC}"
cat <<EOF > dailybackup
00 5 * * * $(which tar) -czf ${today}.backup.tar.gz ${RSNAPSHOT_BACKUP_DIR}
EOF

sudo mv dailybackup /etc/cron.d/
}

# Echo out some post install information
post_install() {
  echo ""
  echo "-------------------------------------------------------------------------------"
  echo -e "${DIALOG}Installation and configuration of rsnapshot and s3cmd has completed.${NC}\n"
  echo -e "${DIALOG}Important file locations:${NC}"
  echo -e "Rsnapshot Configuration File - ${LINKY}/etc/rsnapshot.conf${NC}"
  echo -e "Rsnapshot Cronjob File - ${LINKY}/etc/rsnapshot.conf${NC}"
  echo -e "Rsnapshot Configuration File - ${LINKY}/etc/cron.d/rsnapshot${NC}\n"
  echo -e "${DIALOG}Please note that by default this script only configures rsnapshot to backup this system.${NC}" 
  echo -e "${DIALOG}If you would like to add remote systems for rsnapshot to also backup, you will need to edit the ${LINKY}/etc/rsnapshot.conf file.${NC}\n"
  echo -e "${DIALOG}The following guide should assist in setting up remote hosts in rsnapshot: ${LINKY}https://github.com/greyhoundforty/COSTooling/blob/master/rsnapshot.md${NC}\n"
  echo -e "${DIALOG}This script also installed a basic daily cron job to compress the RSNAPSHOT_BACKUP_DIR, timestamp the backup with todays date and send it to Cloud Object Storage."
  echo -e "${DIALOG}To update when this process runs please edit the file: ${LINKY}/etc/cron.d/dailybackup${NC}"
  echo "-------------------------------------------------------------------------------"
}


set_install_variables
install_tools
configure_rsnapshot
configure_s3cmd
cos_backup_schedule
post_install