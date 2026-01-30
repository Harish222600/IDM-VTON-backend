require('dotenv').config();
const app = require('./app');
const connectDB = require('./config/db');

const PORT = process.env.PORT || 5000;

// Connect to database and start server
const startServer = async () => {
    try {
        // Connect to MongoDB
        await connectDB();

        // Start server - bind to 0.0.0.0 to accept connections from mobile devices
        app.listen(PORT, '0.0.0.0', () => {
            console.log(`
╔════════════════════════════════════════════════════════╗
║                                                        ║
║   🚀 Virtual Try-On Backend Server                    ║
║                                                        ║
║   Server running on port ${PORT}                         ║
║   Environment: ${process.env.NODE_ENV || 'development'}                       ║
║                                                        ║
║   API Endpoints:                                       ║
║   • Auth:     /api/auth                               ║
║   • Users:    /api/users                              ║
║   • Garments: /api/garments                           ║
║   • Try-On:   /api/tryon                              ║
║   • Admin:    /api/admin                              ║
║                                                        ║
║   Health:     /health                                  ║
║                                                        ║
╚════════════════════════════════════════════════════════╝
      `);
        });
    } catch (error) {
        console.error('Failed to start server:', error.message);
        process.exit(1);
    }
};

// Handle unhandled promise rejections
process.on('unhandledRejection', (err) => {
    console.error('Unhandled Rejection:', err.message);
    process.exit(1);
});

// Handle uncaught exceptions
process.on('uncaughtException', (err) => {
    console.error('Uncaught Exception:', err.message);
    process.exit(1);
});

startServer();
