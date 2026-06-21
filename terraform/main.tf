terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Deep Learning AMI (GPU, PyTorch) - has CUDA/drivers preinstalled
data "aws_ami" "dlami" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning AMI*PyTorch*Ubuntu*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

resource "aws_security_group" "llm_serving_sg" {
  name        = "llm-deployment-mechanics-sg"
  description = "Allow SSH and inference API ports"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.my_ip_cidr]
  }

  ingress {
    description = "vLLM / TGI API"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = [var.my_ip_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Project = "llm-deployment-mechanics"
  }
}

resource "aws_spot_instance_request" "gpu_box" {
  ami                    = data.aws_ami.dlami.id
  instance_type          = var.instance_type
  spot_price             = var.max_spot_price
  wait_for_fulfillment   = true
  vpc_security_group_ids = [aws_security_group.llm_serving_sg.id]
  key_name               = var.key_pair_name

  root_block_device {
    volume_size = 100 # quantized + base model weights need room
    volume_type = "gp3"
  }

  tags = {
    Name    = "llm-deployment-mechanics-gpu"
    Project = "llm-deployment-mechanics"
  }
}

output "instance_public_ip" {
  value = aws_spot_instance_request.gpu_box.public_ip
}

output "ssh_command" {
  value = "ssh -i ${var.key_pair_name}.pem ubuntu@${aws_spot_instance_request.gpu_box.public_ip}"
}
