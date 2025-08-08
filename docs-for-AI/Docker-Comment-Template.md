# Docker Comment Template for Professional Container Documentation

This template provides a comprehensive structure for documenting Dockerfiles with clear, maintainable, and professional comments. It follows similar principles to code documentation templates but is specifically designed for Docker container configurations.

## Core Principles

1. **Comprehensive Documentation**: Every Dockerfile should be self-documenting
2. **Structured Organization**: Consistent section organization across all Dockerfiles
3. **Professional Standards**: Clear, technical language appropriate for DevOps teams
4. **Maintenance Focus**: Information needed for updates, debugging, and deployment
5. **Security Awareness**: Document security considerations and best practices

## Template Structure

```dockerfile
# -------------------------------------------------------------------------
# Dockerfile: [dockerfile_name] (e.g., Dockerfile, Dockerfile.frontend, Dockerfile.prod)
# Project: [Project Name] - [Brief Project Description]
# Author: [Author Name]
# Date: [YYYY-MM-DD] (Creation Date)
# Last Modified: [YYYY-MM-DD]
# File Path: [relative_path_to_dockerfile] (e.g., ./Dockerfile, ./frontend/Dockerfile)
# ------------------------------------------------------------------------

# --- Image Objective ---
# [A concise, high-level summary of what this Docker image provides.]
# [Explain its purpose, the main services it runs, and its role within the larger
# system architecture. Include the target environment and use case.]
# Example: "This image provides a production-ready Node.js web application server
# with nginx reverse proxy for the MRCA frontend, optimized for containerized
# deployment with health monitoring and security hardening."
# -------------------------------------------------------------------------

# --- Image Contents Overview ---
# [A brief, bulleted list of the primary components, services, and tools
# included in this image. This helps understand what's running inside.]
# - Base OS: [operating_system_and_version]
# - Runtime: [runtime_environment] (e.g., Node.js 18, Python 3.11, nginx 1.24)
# - Application: [main_application_or_service]
# - Dependencies: [key_dependencies_or_packages]
# - Services: [services_that_will_run] (e.g., web server, database, cache)
# - Tools: [additional_tools_or_utilities]
# -------------------------------------------------------------------------

# --- Dependencies / Base Images ---
# [List and explain the base images and external dependencies this Dockerfile relies on.]
# - Base Image: [base_image_name:tag] - [reason_for_choice_and_features]
# - External Images: [any_multi-stage_build_images] 
# - Package Managers: [apt, yum, apk, npm, pip, etc.] - [packages_being_installed]
# - External Resources: [any_downloaded_files_or_remote_dependencies]
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# [Explain how this Docker image is intended to be built, run, and integrated
# into the larger system. Include common deployment patterns.]
# Build Command: docker build -t [image_name]:[tag] [build_context]
# Run Command: docker run [options] [image_name]:[tag]
# Docker Compose: Part of [compose_file] as service [service_name]
# Deployment: [how_this_fits_into_deployment_pipeline]
# Environment: [target_environments] (development, staging, production)
# -------------------------------------------------------------------------

# --- Build Arguments & Environment Variables ---
# [Document all configurable build arguments and runtime environment variables.]
# Build Arguments (ARG):
#   - [ARG_NAME]: [description] (default: [default_value])
# Environment Variables (ENV):
#   - [ENV_VAR]: [description] (default: [default_value])
#   - [REQUIRED_ENV]: [description] (required at runtime)
# -------------------------------------------------------------------------

# --- Ports & Volumes ---
# [Document all exposed ports and volume mount points.]
# Exposed Ports:
#   - [port_number]/[protocol]: [service_description]
# Volumes:
#   - [volume_path]: [purpose_and_data_type]
# Health Check: [if_implemented] on port [port] endpoint [endpoint]
# -------------------------------------------------------------------------

# --- Security Considerations ---
# [Document security measures, user permissions, and security best practices.]
# - User: Runs as [user_name] (UID: [uid]) for security
# - Permissions: [file_and_directory_permissions]
# - Secrets: [how_secrets_are_handled]
# - Network: [network_security_considerations]
# - Vulnerabilities: [known_issues_or_mitigations]
# -------------------------------------------------------------------------

# --- Apache-2.0 ---
# Copyright [YEAR] [Copyright Holder Name]
# SPDX-License-Identifier: Apache-2.0
# -------------------------------------------------------------------------

# =========================================================================
# Multi-Stage Build Setup (if applicable)
# =========================================================================
# Stage 1: [stage_name] - [purpose_of_this_stage]
# This stage [explains_what_this_stage_does_and_why]

# Base image selection with version pinning for reproducibility
FROM [base_image]:[specific_version] AS [stage_name]

# Document the base image choice and version strategy
# [base_image]:[version] chosen for [reasons: security, size, compatibility, LTS, etc.]

# =========================================================================
# Build Arguments Declaration
# =========================================================================
# Declare build-time variables that can be passed during docker build

# [ARG_NAME] - [Description of what this argument controls]
ARG [ARG_NAME]=[default_value]

# Example patterns:
# ARG NODE_VERSION=18
# ARG BUILD_ENV=production
# ARG USER_ID=1000

# =========================================================================
# Environment Variables
# =========================================================================
# Set environment variables for runtime configuration

# [ENV_VAR] - [Description of what this variable controls]
ENV [ENV_VAR]=[value]

# Example patterns:
# ENV NODE_ENV=production
# ENV PORT=3000
# ENV LOG_LEVEL=info

# =========================================================================
# System Package Installation
# =========================================================================
# Install system-level dependencies and packages

# Update package manager and install essential packages
# [Explain why each package is needed and group related packages]
RUN [package_manager] update \
    && [package_manager] install -y \
        [package1] \          # [Purpose: reason for this package]
        [package2] \          # [Purpose: reason for this package]
        [package3] \          # [Purpose: reason for this package]
    && [package_manager] clean \
    && rm -rf [cache_directories]

# Example patterns:
# RUN apt-get update \
#     && apt-get install -y \
#         curl \              # For health checks and external API calls
#         wget \              # For downloading external resources
#         ca-certificates \   # For SSL/TLS certificate validation
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# =========================================================================
# User and Security Setup
# =========================================================================
# Create non-root user for security best practices

# Create application user with specific UID for consistency across environments
# This prevents running as root and provides better security isolation
RUN groupadd -g [GID] [groupname] \
    && useradd -u [UID] -g [GID] -m -s /bin/bash [username]

# Example:
# RUN groupadd -g 1000 appuser \
#     && useradd -u 1000 -g 1000 -m -s /bin/bash appuser

# =========================================================================
# Application Dependencies Installation
# =========================================================================
# Install application-specific dependencies (language packages, libraries)

# Set working directory for dependency installation
WORKDIR [dependency_install_directory]

# Copy dependency files first for better layer caching
# This allows Docker to cache dependency installation when only source code changes
COPY [dependency_files] [destination]

# Example patterns for different languages:
# Node.js:
# COPY package*.json ./
# RUN npm ci --only=production

# Python:
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

# Install dependencies with caching optimization
RUN [dependency_install_command] \
    && [cleanup_commands]

# =========================================================================
# Application Code Setup
# =========================================================================
# Copy application source code and configure the application environment

# Set the final working directory for the application
WORKDIR [app_directory]

# Copy application source code
# Use .dockerignore to exclude unnecessary files (node_modules, .git, etc.)
COPY [source_files] [destination]

# Change ownership to application user for security
RUN chown -R [username]:[groupname] [app_directory]

# =========================================================================
# Application Build Process (if applicable)
# =========================================================================
# Build the application if compilation/transpilation is required

# Switch to application user for build process
USER [username]

# Build application with proper environment
RUN [build_commands] \
    && [post_build_cleanup]

# Example patterns:
# RUN npm run build
# RUN python setup.py build
# RUN go build -o app main.go

# =========================================================================
# Runtime Configuration
# =========================================================================
# Configure the container for runtime execution

# Expose application ports
# [port] - [service description and protocol]
EXPOSE [port]

# Example:
# EXPOSE 3000    # HTTP API server
# EXPOSE 9090    # Metrics endpoint

# Configure volume mount points for persistent data
# [volume_path] - [description of data stored]
VOLUME ["[volume_path]"]

# Example:
# VOLUME ["/app/data"]     # Application data storage
# VOLUME ["/app/logs"]     # Application log files

# =========================================================================
# Health Check Configuration
# =========================================================================
# Configure container health monitoring

# Health check ensures the container is responding correctly
# --interval: How often to check
# --timeout: How long to wait for response
# --start-period: Grace period before first check
# --retries: Number of consecutive failures before marking unhealthy
HEALTHCHECK --interval=[interval] --timeout=[timeout] --start-period=[start_period] --retries=[retries] \
    CMD [health_check_command] || exit 1

# Example:
# HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
#     CMD curl -f http://localhost:3000/health || exit 1

# =========================================================================
# Startup Configuration
# =========================================================================
# Configure how the container starts and runs

# Switch to non-root user for security (if not already done)
USER [username]

# Set the default command to run when container starts
# Use ENTRYPOINT for fixed command + CMD for default arguments pattern
# Or use CMD for simple command execution

# ENTRYPOINT + CMD pattern (recommended for production):
ENTRYPOINT ["[entrypoint_script_or_command]"]
CMD ["[default_arguments]"]

# Or simple CMD pattern:
CMD ["[command]", "[arg1]", "[arg2]"]

# Example patterns:
# ENTRYPOINT ["docker-entrypoint.sh"]
# CMD ["node", "server.js"]

# Or:
# CMD ["python", "-m", "app.main"]

# =========================================================================
# Multi-Stage Build - Production Stage (if applicable)
# =========================================================================
# Stage 2: [production_stage_name] - [purpose_of_production_stage]
# This stage [explains_production_optimizations]

# Use distroless or minimal base image for production
FROM [minimal_base_image]:[version] AS [production_stage]

# Copy only necessary artifacts from build stage
COPY --from=[build_stage] [source_path] [destination_path]

# Repeat security, runtime, and startup configuration for production stage
# [Include relevant sections from above adapted for production]

# =========================================================================
# Build Instructions / Usage Examples
# =========================================================================
# 
# Build Commands:
#   Development: docker build -t [image_name]:dev .
#   Production:  docker build -t [image_name]:prod --target [production_stage] .
#   With args:   docker build -t [image_name] --build-arg [ARG]=[value] .
#
# Run Commands:
#   Basic:       docker run -p [host_port]:[container_port] [image_name]
#   With env:    docker run -e [ENV_VAR]=[value] [image_name]
#   With volume: docker run -v [host_path]:[container_path] [image_name]
#
# Docker Compose:
#   services:
#     [service_name]:
#       build: [build_context]
#       ports:
#         - "[host_port]:[container_port]"
#       environment:
#         - [ENV_VAR]=[value]
#       volumes:
#         - [volume_mapping]
#
# =========================================================================
# End of Dockerfile
# =========================================================================
```

## Key Sections Explained

### 1. **File Header**
- Project information and maintainer details
- Creation and modification dates for change tracking
- File path for easy location in complex projects

### 2. **Image Objective**
- Clear purpose statement for the container
- Target environment and use case
- Role within larger system architecture

### 3. **Contents Overview**
- Quick reference of what's included in the image
- Base OS, runtime, and key dependencies
- Services and tools that will be running

### 4. **Dependencies Documentation**
- Base image selection rationale
- External dependencies and package requirements
- Security and compatibility considerations

### 5. **Usage Integration**
- Build and run commands with examples
- Integration with Docker Compose and orchestration
- Deployment pipeline considerations

### 6. **Configuration Documentation**
- All build arguments and environment variables
- Default values and required configurations
- Runtime configuration options

### 7. **Ports and Volumes**
- All exposed ports with service descriptions
- Volume mount points and data persistence
- Health check endpoints

### 8. **Security Considerations**
- User permissions and non-root execution
- Security best practices implementation
- Known vulnerabilities or mitigations

## Implementation Guidelines

### For Single-Stage Builds:
- Use the template sections linearly
- Remove multi-stage specific comments
- Focus on runtime optimization within single stage

### For Multi-Stage Builds:
- Document each stage clearly
- Explain the purpose of stage separation
- Show artifact copying between stages

### For Different Application Types:
- **Web Applications**: Focus on port exposure, health checks, static file serving
- **Databases**: Emphasize data volumes, initialization scripts, backup procedures
- **Microservices**: Document service discovery, communication ports, dependencies
- **Build Tools**: Focus on build artifacts, caching strategies, output volumes

## Best Practices for Docker Documentation

1. **Version Pinning**: Always document why specific versions are chosen
2. **Layer Optimization**: Explain caching strategies and layer organization
3. **Security**: Document user permissions, secret handling, and security measures
4. **Multi-Environment**: Show how the same Dockerfile works across environments
5. **Troubleshooting**: Include common build/run issues and solutions
6. **Performance**: Document resource requirements and optimization strategies

This template ensures that Docker configurations are as well-documented and maintainable as source code, making them suitable for professional development environments and production deployments. 