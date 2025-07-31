# ML-TA Infrastructure as Code
# This Terraform configuration creates the complete AWS infrastructure for ML-TA system

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }

  backend "s3" {
    bucket = "ml-ta-terraform-state"
    key    = "production/terraform.tfstate"
    region = "us-west-2"
    
    dynamodb_table = "ml-ta-terraform-locks"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "ML-TA"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# VPC and Networking
resource "aws_vpc" "ml_ta_vpc" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.project_name}-vpc"
  }
}

resource "aws_internet_gateway" "ml_ta_igw" {
  vpc_id = aws_vpc.ml_ta_vpc.id

  tags = {
    Name = "${var.project_name}-igw"
  }
}

resource "aws_subnet" "public_subnets" {
  count = length(var.public_subnet_cidrs)

  vpc_id                  = aws_vpc.ml_ta_vpc.id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project_name}-public-subnet-${count.index + 1}"
    Type = "Public"
  }
}

resource "aws_subnet" "private_subnets" {
  count = length(var.private_subnet_cidrs)

  vpc_id            = aws_vpc.ml_ta_vpc.id
  cidr_block        = var.private_subnet_cidrs[count.index]
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "${var.project_name}-private-subnet-${count.index + 1}"
    Type = "Private"
  }
}

resource "aws_route_table" "public_rt" {
  vpc_id = aws_vpc.ml_ta_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.ml_ta_igw.id
  }

  tags = {
    Name = "${var.project_name}-public-rt"
  }
}

resource "aws_route_table_association" "public_rta" {
  count = length(aws_subnet.public_subnets)

  subnet_id      = aws_subnet.public_subnets[count.index].id
  route_table_id = aws_route_table.public_rt.id
}

# NAT Gateway for private subnets
resource "aws_eip" "nat_eip" {
  count = length(aws_subnet.public_subnets)

  domain = "vpc"
  depends_on = [aws_internet_gateway.ml_ta_igw]

  tags = {
    Name = "${var.project_name}-nat-eip-${count.index + 1}"
  }
}

resource "aws_nat_gateway" "nat_gw" {
  count = length(aws_subnet.public_subnets)

  allocation_id = aws_eip.nat_eip[count.index].id
  subnet_id     = aws_subnet.public_subnets[count.index].id

  tags = {
    Name = "${var.project_name}-nat-gw-${count.index + 1}"
  }
}

resource "aws_route_table" "private_rt" {
  count = length(aws_subnet.private_subnets)

  vpc_id = aws_vpc.ml_ta_vpc.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat_gw[count.index].id
  }

  tags = {
    Name = "${var.project_name}-private-rt-${count.index + 1}"
  }
}

resource "aws_route_table_association" "private_rta" {
  count = length(aws_subnet.private_subnets)

  subnet_id      = aws_subnet.private_subnets[count.index].id
  route_table_id = aws_route_table.private_rt[count.index].id
}

# Security Groups
resource "aws_security_group" "eks_cluster_sg" {
  name_prefix = "${var.project_name}-eks-cluster-"
  vpc_id      = aws_vpc.ml_ta_vpc.id

  ingress {
    from_port = 443
    to_port   = 443
    protocol  = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-eks-cluster-sg"
  }
}

resource "aws_security_group" "eks_node_sg" {
  name_prefix = "${var.project_name}-eks-node-"
  vpc_id      = aws_vpc.ml_ta_vpc.id

  ingress {
    from_port = 0
    to_port   = 65535
    protocol  = "tcp"
    self      = true
  }

  ingress {
    from_port       = 1025
    to_port         = 65535
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_cluster_sg.id]
  }

  ingress {
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_cluster_sg.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-eks-node-sg"
  }
}

resource "aws_security_group" "rds_sg" {
  name_prefix = "${var.project_name}-rds-"
  vpc_id      = aws_vpc.ml_ta_vpc.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_node_sg.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-rds-sg"
  }
}

resource "aws_security_group" "redis_sg" {
  name_prefix = "${var.project_name}-redis-"
  vpc_id      = aws_vpc.ml_ta_vpc.id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_node_sg.id]
  }

  tags = {
    Name = "${var.project_name}-redis-sg"
  }
}

# EKS Cluster
resource "aws_eks_cluster" "ml_ta_cluster" {
  name     = var.cluster_name
  role_arn = aws_iam_role.eks_cluster_role.arn
  version  = var.kubernetes_version

  vpc_config {
    subnet_ids              = concat(aws_subnet.public_subnets[*].id, aws_subnet.private_subnets[*].id)
    security_group_ids      = [aws_security_group.eks_cluster_sg.id]
    endpoint_private_access = true
    endpoint_public_access  = true
    public_access_cidrs     = var.cluster_endpoint_public_access_cidrs
  }

  encryption_config {
    provider {
      key_arn = aws_kms_key.eks_encryption_key.arn
    }
    resources = ["secrets"]
  }

  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
    aws_iam_role_policy_attachment.eks_vpc_resource_controller,
    aws_cloudwatch_log_group.eks_cluster_logs
  ]

  tags = {
    Name = var.cluster_name
  }
}

# EKS Node Groups
resource "aws_eks_node_group" "ml_ta_nodes" {
  cluster_name    = aws_eks_cluster.ml_ta_cluster.name
  node_group_name = "${var.project_name}-nodes"
  node_role_arn   = aws_iam_role.eks_node_role.arn
  subnet_ids      = aws_subnet.private_subnets[*].id

  capacity_type  = "ON_DEMAND"
  instance_types = var.node_instance_types

  scaling_config {
    desired_size = var.node_desired_size
    max_size     = var.node_max_size
    min_size     = var.node_min_size
  }

  update_config {
    max_unavailable_percentage = 25
  }

  ami_type       = "AL2_x86_64"
  disk_size      = var.node_disk_size
  
  labels = {
    Environment = var.environment
    NodeGroup   = "ml-ta-nodes"
  }

  taint {
    key    = "ml-workload"
    value  = "true"
    effect = "NO_SCHEDULE"
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.eks_container_registry_policy,
  ]

  tags = {
    Name = "${var.project_name}-node-group"
  }
}

# RDS PostgreSQL Database
resource "aws_db_subnet_group" "ml_ta_db_subnet_group" {
  name       = "${var.project_name}-db-subnet-group"
  subnet_ids = aws_subnet.private_subnets[*].id

  tags = {
    Name = "${var.project_name}-db-subnet-group"
  }
}

resource "aws_db_instance" "ml_ta_database" {
  identifier = "${var.project_name}-database"

  engine         = "postgres"
  engine_version = var.postgres_version
  instance_class = var.db_instance_class

  allocated_storage     = var.db_allocated_storage
  max_allocated_storage = var.db_max_allocated_storage
  storage_type          = "gp3"
  storage_encrypted     = true
  kms_key_id           = aws_kms_key.rds_encryption_key.arn

  db_name  = var.database_name
  username = var.database_username
  password = var.database_password

  vpc_security_group_ids = [aws_security_group.rds_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.ml_ta_db_subnet_group.name

  backup_retention_period = var.db_backup_retention_period
  backup_window          = var.db_backup_window
  maintenance_window     = var.db_maintenance_window

  multi_az               = var.environment == "production" ? true : false
  publicly_accessible   = false
  deletion_protection    = var.environment == "production" ? true : false
  skip_final_snapshot    = var.environment != "production"
  final_snapshot_identifier = var.environment == "production" ? "${var.project_name}-final-snapshot-${formatdate("YYYY-MM-DD-hhmm", timestamp())}" : null

  performance_insights_enabled = true
  monitoring_interval         = 60
  monitoring_role_arn        = aws_iam_role.rds_monitoring_role.arn

  tags = {
    Name = "${var.project_name}-database"
  }
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "ml_ta_redis_subnet_group" {
  name       = "${var.project_name}-redis-subnet-group"
  subnet_ids = aws_subnet.private_subnets[*].id
}

resource "aws_elasticache_replication_group" "ml_ta_redis" {
  replication_group_id       = "${var.project_name}-redis"
  description                = "Redis cluster for ML-TA caching"

  node_type            = var.redis_node_type
  port                 = 6379
  parameter_group_name = "default.redis7"

  num_cache_clusters = var.redis_num_cache_nodes
  
  subnet_group_name  = aws_elasticache_subnet_group.ml_ta_redis_subnet_group.name
  security_group_ids = [aws_security_group.redis_sg.id]

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = var.redis_auth_token

  snapshot_retention_limit = 3
  snapshot_window         = "03:00-05:00"
  maintenance_window      = "sun:05:00-sun:07:00"

  tags = {
    Name = "${var.project_name}-redis"
  }
}

# S3 Buckets
resource "aws_s3_bucket" "ml_ta_models" {
  bucket = "${var.project_name}-models-${random_string.bucket_suffix.result}"

  tags = {
    Name        = "${var.project_name}-models"
    Purpose     = "ML Model Storage"
  }
}

resource "aws_s3_bucket" "ml_ta_data" {
  bucket = "${var.project_name}-data-${random_string.bucket_suffix.result}"

  tags = {
    Name        = "${var.project_name}-data"
    Purpose     = "Data Storage"
  }
}

resource "aws_s3_bucket" "ml_ta_logs" {
  bucket = "${var.project_name}-logs-${random_string.bucket_suffix.result}"

  tags = {
    Name        = "${var.project_name}-logs"
    Purpose     = "Log Storage"
  }
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

# S3 Bucket Configurations
resource "aws_s3_bucket_versioning" "ml_ta_models_versioning" {
  bucket = aws_s3_bucket.ml_ta_models.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "ml_ta_models_encryption" {
  bucket = aws_s3_bucket.ml_ta_models.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.s3_encryption_key.arn
      sse_algorithm     = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "ml_ta_models_pab" {
  bucket = aws_s3_bucket.ml_ta_models.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Similar configurations for other buckets...
resource "aws_s3_bucket_versioning" "ml_ta_data_versioning" {
  bucket = aws_s3_bucket.ml_ta_data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "ml_ta_data_encryption" {
  bucket = aws_s3_bucket.ml_ta_data.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.s3_encryption_key.arn
      sse_algorithm     = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "ml_ta_data_pab" {
  bucket = aws_s3_bucket.ml_ta_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "eks_cluster_logs" {
  name              = "/aws/eks/${var.cluster_name}/cluster"
  retention_in_days = var.log_retention_days
  kms_key_id        = aws_kms_key.cloudwatch_encryption_key.arn

  tags = {
    Name = "${var.project_name}-eks-cluster-logs"
  }
}

resource "aws_cloudwatch_log_group" "ml_ta_application_logs" {
  name              = "/aws/ml-ta/application"
  retention_in_days = var.log_retention_days
  kms_key_id        = aws_kms_key.cloudwatch_encryption_key.arn

  tags = {
    Name = "${var.project_name}-application-logs"
  }
}

# Application Load Balancer
resource "aws_lb" "ml_ta_alb" {
  name               = "${var.project_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = aws_subnet.public_subnets[*].id

  enable_deletion_protection = var.environment == "production" ? true : false

  access_logs {
    bucket  = aws_s3_bucket.ml_ta_logs.bucket
    prefix  = "alb-logs"
    enabled = true
  }

  tags = {
    Name = "${var.project_name}-alb"
  }
}

resource "aws_security_group" "alb_sg" {
  name_prefix = "${var.project_name}-alb-"
  vpc_id      = aws_vpc.ml_ta_vpc.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-alb-sg"
  }
}

# Outputs
output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = aws_eks_cluster.ml_ta_cluster.endpoint
}

output "cluster_security_group_id" {
  description = "Security group ids attached to the cluster control plane"
  value       = aws_eks_cluster.ml_ta_cluster.vpc_config[0].cluster_security_group_id
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = aws_eks_cluster.ml_ta_cluster.name
}

output "database_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.ml_ta_database.endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = aws_elasticache_replication_group.ml_ta_redis.primary_endpoint_address
  sensitive   = true
}

output "s3_models_bucket" {
  description = "S3 bucket for ML models"
  value       = aws_s3_bucket.ml_ta_models.bucket
}

output "s3_data_bucket" {
  description = "S3 bucket for data"
  value       = aws_s3_bucket.ml_ta_data.bucket
}

output "load_balancer_dns" {
  description = "DNS name of the load balancer"
  value       = aws_lb.ml_ta_alb.dns_name
}
