#This is a sample project to explain how to deploy a machine learning model for sentiment prediction on aws ec2 instance using docker image.

#installing docker in aws ec2 instance
sudo dnf install docker -y

#starting docker service
sudo service docker start

#provide access to ec2-user to docker
 sudo usermod -a -G docker ec2-user

#Build docker image
 docker build -t my-docker-image .
 
![image](https://github.com/user-attachments/assets/ecebe104-3d5f-492c-a166-76ab2569682d)

 ![image](https://github.com/user-attachments/assets/1ef94fff-9edb-4fa3-8ad9-d233a20a5c20)


#Check status of docker image
docker images

![image](https://github.com/user-attachments/assets/f7b26518-58f1-48b6-a8a7-036d40e8f815)

#Running docker image

docker run -p 80:80 my-docker-image

![image](https://github.com/user-attachments/assets/5eeb412c-d973-4aaf-ad60-b2160360dfe7)


#Viewing the sentiment analysis app on browser.
#Copy the public ip address from aws console and copy in the browser url.

![image](https://github.com/user-attachments/assets/2194d617-e5b8-457c-8eb1-4098a3593955)

ec2-51-20-9-189.eu-north-1.compute.amazonaws.com 

![image](https://github.com/user-attachments/assets/4cd68fec-ead3-4f68-8fbf-21422fec3073)


# Running the sentiment analysis with command line
http://ec2-13-60-37-27.eu-north-1.compute.amazonaws.com/sentiment?Review=%22movie%20was%20boring%22


![image](https://github.com/user-attachments/assets/d33aa57d-ec27-49c1-a13c-22dc1a8cdd2d)
