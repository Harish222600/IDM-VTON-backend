# Virtual Try-On Backend

## Setup

1. Install dependencies:
```bash
npm install
```

2. Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```

3. Start MongoDB locally or use MongoDB Atlas.

4. Create a Supabase project and storage bucket named `tryon-images`.

5. Get a Hugging Face API key with access to IDM-VTON model.

## Running

Development mode:
```bash
npm run dev
```

Production mode:
```bash
npm start
```

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login
- `POST /api/auth/forgot-password` - Request password reset
- `POST /api/auth/reset-password` - Reset password
- `POST /api/auth/logout` - Logout
- `GET /api/auth/me` - Get current user

### Users (Protected)
- `GET /api/users/profile` - Get profile
- `PUT /api/users/profile` - Update profile
- `PUT /api/users/profile-image` - Upload profile image
- `PUT /api/users/password` - Change password
- `DELETE /api/users/account` - Delete account
- `GET /api/users/favorites` - Get favorites
- `POST /api/users/favorites/:id` - Add to favorites
- `DELETE /api/users/favorites/:id` - Remove from favorites

### Garments (Public)
- `GET /api/garments` - Get all garments (with filters)
- `GET /api/garments/:id` - Get single garment
- `GET /api/garments/categories` - Get categories
- `GET /api/garments/colors` - Get colors

### Try-On (Protected)
- `POST /api/tryon` - Initiate try-on
- `GET /api/tryon/history` - Get history
- `GET /api/tryon/:id` - Get result
- `DELETE /api/tryon/:id` - Delete result
- `DELETE /api/tryon/history` - Clear history

### Admin (Admin Only)
- `GET /api/admin/users` - Get all users
- `PUT /api/admin/users/:id/block` - Block/unblock user
- `DELETE /api/admin/users/:id` - Delete user
- `GET /api/admin/users/:id/activity` - Get user activity
- `GET /api/admin/garments` - Get all garments
- `POST /api/admin/garments` - Create garment
- `PUT /api/admin/garments/:id` - Update garment
- `DELETE /api/admin/garments/:id` - Delete garment
- `GET /api/admin/analytics` - Get analytics
- `GET /api/admin/logs` - Get system logs
