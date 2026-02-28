# Deployment Guide

This guide covers deploying Misa to various platforms and environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Configuration](#environment-configuration)
- [Local Production Build](#local-production-build)
- [Vercel Deployment (Frontend)](#vercel-deployment-frontend)
- [Heroku Deployment (Backend)](#heroku-deployment-backend)
- [AWS Deployment](#aws-deployment)
- [Docker Deployment](#docker-deployment)
- [Monitoring & Maintenance](#monitoring--maintenance)

## Prerequisites

- Node.js 16+ installed
- Git repository access
- Gemini API key
- Platform-specific accounts (Vercel, Heroku, AWS, etc.)

## Environment Configuration

### Frontend Environment Variables

Create `.env.production`:

```env
VITE_API_URL=https://your-backend-domain.com/api
VITE_GEMINI_API_KEY=your_gemini_api_key
```

### Backend Environment Variables

Create `.env.production`:

```env
NODE_ENV=production
PORT=5000
GEMINI_API_KEY=your_gemini_api_key
ALLOWED_ORIGINS=https://your-frontend-domain.com
NOAA_API_KEY=your_noaa_key
MOVEBANK_USERNAME=your_username
MOVEBANK_PASSWORD=your_password
```

## Local Production Build

### Build Frontend

```bash
# Install dependencies
npm install

# Build for production
npm run build

# Preview production build
npm run preview
```

The build output will be in the `dist/` directory.

### Build Backend

```bash
cd backend

# Install production dependencies
npm install --production

# Start production server
NODE_ENV=production npm start
```

## Vercel Deployment (Frontend)

### Option 1: Vercel CLI

1. **Install Vercel CLI**
```bash
npm install -g vercel
```

2. **Login to Vercel**
```bash
vercel login
```

3. **Deploy**
```bash
vercel --prod
```

4. **Set Environment Variables**
```bash
vercel env add VITE_API_URL
vercel env add VITE_GEMINI_API_KEY
```

### Option 2: GitHub Integration

1. Push code to GitHub
2. Go to [vercel.com](https://vercel.com)
3. Click "Import Project"
4. Select your repository
5. Configure:
   - **Framework Preset**: Vite
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
6. Add environment variables in Vercel dashboard
7. Deploy

### Vercel Configuration

Create `vercel.json`:

```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "framework": "vite",
  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "/index.html"
    }
  ]
}
```

## Heroku Deployment (Backend)

### Setup

1. **Install Heroku CLI**
```bash
npm install -g heroku
```

2. **Login**
```bash
heroku login
```

3. **Create App**
```bash
cd backend
heroku create misa-backend
```

4. **Set Environment Variables**
```bash
heroku config:set GEMINI_API_KEY=your_key
heroku config:set NODE_ENV=production
heroku config:set ALLOWED_ORIGINS=https://your-frontend.vercel.app
```

5. **Create Procfile**

Create `backend/Procfile`:
```
web: node src/server.js
```

6. **Deploy**
```bash
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

7. **Scale Dynos**
```bash
heroku ps:scale web=1
```

### Heroku Configuration

Update `backend/package.json`:

```json
{
  "engines": {
    "node": "16.x",
    "npm": "8.x"
  },
  "scripts": {
    "start": "node src/server.js"
  }
}
```

## AWS Deployment

### AWS Elastic Beanstalk (Backend)

1. **Install EB CLI**
```bash
pip install awsebcli
```

2. **Initialize**
```bash
cd backend
eb init -p node.js-16 misa-backend
```

3. **Create Environment**
```bash
eb create misa-backend-prod
```

4. **Set Environment Variables**
```bash
eb setenv GEMINI_API_KEY=your_key NODE_ENV=production
```

5. **Deploy**
```bash
eb deploy
```

### AWS S3 + CloudFront (Frontend)

1. **Build Frontend**
```bash
npm run build
```

2. **Create S3 Bucket**
```bash
aws s3 mb s3://misa-frontend
```

3. **Upload Build**
```bash
aws s3 sync dist/ s3://misa-frontend --delete
```

4. **Configure S3 for Static Hosting**
```bash
aws s3 website s3://misa-frontend --index-document index.html
```

5. **Create CloudFront Distribution**
- Origin: S3 bucket
- Default Root Object: index.html
- Error Pages: 404 → /index.html (for SPA routing)

### AWS EC2 (Full Stack)

1. **Launch EC2 Instance**
   - AMI: Ubuntu 22.04
   - Instance Type: t2.micro (free tier)
   - Security Group: Allow ports 22, 80, 443, 5000

2. **Connect to Instance**
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

3. **Install Node.js**
```bash
curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -
sudo apt-get install -y nodejs
```

4. **Install Nginx**
```bash
sudo apt-get install nginx
```

5. **Clone Repository**
```bash
git clone your-repo-url
cd Misa
```

6. **Setup Backend**
```bash
cd backend
npm install
cp .env.example .env
# Edit .env with production values
```

7. **Setup PM2**
```bash
sudo npm install -g pm2
pm2 start src/server.js --name misa-backend
pm2 startup
pm2 save
```

8. **Build Frontend**
```bash
cd ..
npm install
npm run build
```

9. **Configure Nginx**

Create `/etc/nginx/sites-available/misa`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Frontend
    location / {
        root /home/ubuntu/Misa/dist;
        try_files $uri $uri/ /index.html;
    }

    # Backend API
    location /api {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

10. **Enable Site**
```bash
sudo ln -s /etc/nginx/sites-available/misa /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

11. **Setup SSL with Let's Encrypt**
```bash
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## Docker Deployment

### Frontend Dockerfile

Create `Dockerfile`:

```dockerfile
FROM node:16-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

Create `nginx.conf`:

```nginx
server {
    listen 80;
    location / {
        root /usr/share/nginx/html;
        try_files $uri $uri/ /index.html;
    }
}
```

### Backend Dockerfile

Create `backend/Dockerfile`:

```dockerfile
FROM node:16-alpine

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .

EXPOSE 5000
CMD ["node", "src/server.js"]
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  frontend:
    build: .
    ports:
      - "80:80"
    environment:
      - VITE_API_URL=http://backend:5000/api
    depends_on:
      - backend

  backend:
    build: ./backend
    ports:
      - "5000:5000"
    environment:
      - NODE_ENV=production
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - PORT=5000
    restart: unless-stopped
```

### Deploy with Docker

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Monitoring & Maintenance

### Health Checks

**Backend Health Endpoint:**
```bash
curl https://your-backend.com/health
```

Expected response:
```json
{
  "status": "ok",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Logging

**View Heroku Logs:**
```bash
heroku logs --tail
```

**View PM2 Logs:**
```bash
pm2 logs misa-backend
```

**View Docker Logs:**
```bash
docker-compose logs -f backend
```

### Monitoring Tools

- **Uptime Monitoring**: UptimeRobot, Pingdom
- **Error Tracking**: Sentry
- **Performance**: New Relic, DataDog
- **Analytics**: Google Analytics, Plausible

### Backup Strategy

1. **Database Backups** (when implemented)
   - Daily automated backups
   - 30-day retention
   - Off-site storage

2. **Configuration Backups**
   - Environment variables documented
   - Infrastructure as Code (Terraform)

### Scaling

**Horizontal Scaling:**
- Add more backend instances
- Use load balancer (AWS ALB, Nginx)
- Implement session management

**Vertical Scaling:**
- Upgrade instance size
- Increase memory/CPU

**Database Scaling:**
- Read replicas
- Connection pooling
- Query optimization

## Troubleshooting

### Common Issues

**CORS Errors:**
- Check `ALLOWED_ORIGINS` in backend
- Verify frontend URL matches

**API Connection Failed:**
- Check `VITE_API_URL` in frontend
- Verify backend is running
- Check firewall rules

**Build Failures:**
- Clear node_modules: `rm -rf node_modules && npm install`
- Check Node.js version
- Verify environment variables

**Performance Issues:**
- Enable compression
- Implement caching
- Optimize database queries
- Use CDN for static assets

## Rollback Procedure

### Vercel
```bash
vercel rollback
```

### Heroku
```bash
heroku releases
heroku rollback v123
```

### Docker
```bash
docker-compose down
git checkout previous-commit
docker-compose up -d --build
```

## Security Checklist

- [ ] HTTPS enabled
- [ ] Environment variables secured
- [ ] API keys not in code
- [ ] CORS properly configured
- [ ] Security headers enabled (Helmet)
- [ ] Rate limiting implemented
- [ ] Input validation active
- [ ] Dependencies updated
- [ ] Secrets rotation scheduled
- [ ] Monitoring alerts configured

## Post-Deployment

1. Test all features in production
2. Monitor error rates
3. Check performance metrics
4. Verify SSL certificate
5. Test from different locations
6. Update documentation
7. Notify team of deployment

## Support

For deployment issues:
- Check logs first
- Review this documentation
- Open GitHub issue
- Contact DevOps team
