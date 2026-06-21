variable "aws_region" {
  description = "AWS region to deploy in"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "GPU instance type"
  type        = string
  default     = "g5.xlarge" # 1x A10G, 24GB VRAM, 4 vCPU, 16GB RAM
}

variable "max_spot_price" {
  description = "Max hourly price you're willing to pay (g5.xlarge on-demand is ~$1.00/hr, spot usually ~$0.40-0.50/hr)"
  type        = string
  default     = "0.60"
}

variable "key_pair_name" {
  description = "Name of an existing EC2 key pair for SSH access"
  type        = string
}

variable "my_ip_cidr" {
  description = "Your IP in CIDR form, e.g. 1.2.3.4/32 - get it from https://checkip.amazonaws.com"
  type        = string
}
