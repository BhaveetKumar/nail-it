# 🌐 Node.js API Design Complete Guide

> **Master REST, GraphQL, and gRPC API design with Node.js best practices**

## 📚 Overview

This comprehensive guide covers API design principles, patterns, and implementations using Node.js. Learn to build scalable, maintainable, and secure APIs for modern applications.

## 🎯 Table of Contents

1. [REST API Design](#rest-api-design/)
2. [GraphQL API Design](#graphql-api-design/)
3. [gRPC API Design](#grpc-api-design/)
4. [API Security](#api-security/)
5. [API Documentation](#api-documentation/)
6. [API Testing](#api-testing/)
7. [API Performance](#api-performance/)
8. [API Versioning](#api-versioning/)

## 🔗 REST API Design

### **RESTful Principles**

```javascript
// Express.js REST API Structure
const express = require("express");
const app = express();

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// CORS middleware
app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
  res.header(
    "Access-Control-Allow-Headers",
    "Origin, X-Requested-With, Content-Type, Accept, Authorization"
  );

  if (req.method === "OPTIONS") {
    res.sendStatus(200);
  } else {
    next();
  }
});

// API Routes
app.use("/api/v1/users", require("./routes/users"));
app.use("/api/v1/posts", require("./routes/posts"));
app.use("/api/v1/comments", require("./routes/comments"));

module.exports = app;
```

### **Resource-Based URL Design**

```javascript
// Good RESTful URL patterns
const userRoutes = {
  // Collection operations
  "GET /api/v1/users": "List all users",
  "POST /api/v1/users": "Create a new user",

  // Individual resource operations
  "GET /api/v1/users/:id": "Get user by ID",
  "PUT /api/v1/users/:id": "Update user by ID",
  "PATCH /api/v1/users/:id": "Partially update user by ID",
  "DELETE /api/v1/users/:id": "Delete user by ID",

  // Sub-resources
  "GET /api/v1/users/:id/posts": "Get user's posts",
  "GET /api/v1/users/:id/posts/:postId": "Get specific post by user",

  // Actions
  "POST /api/v1/users/:id/activate": "Activate user",
  "POST /api/v1/users/:id/deactivate": "Deactivate user",
};

// Express.js implementation
const express = require("express");
const router = express.Router();

// User controller
class UserController {
  async listUsers(req, res) {
    try {
      const {
        page = 1,
        limit = 10,
        sort = "createdAt",
        order = "desc",
      } = req.query;
      const users = await UserService.getUsers({ page, limit, sort, order });

      res.json({
        success: true,
        data: users,
        pagination: {
          page: parseInt(page),
          limit: parseInt(limit),
          total: users.total,
        },
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        error: error.message,
      });
    }
  }

  async getUser(req, res) {
    try {
      const { id } = req.params;
      const user = await UserService.getUserById(id);

      if (!user) {
        return res.status(404).json({
          success: false,
          error: "User not found",
        });
      }

      res.json({
        success: true,
        data: user,
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        error: error.message,
      });
    }
  }

  async createUser(req, res) {
    try {
      const userData = req.body;
      const user = await UserService.createUser(userData);

      res.status(201).json({
        success: true,
        data: user,
      });
    } catch (error) {
      if (error.name === "ValidationError") {
        return res.status(400).json({
          success: false,
          error: error.message,
          details: error.details,
        });
      }

      res.status(500).json({
        success: false,
        error: error.message,
      });
    }
  }

  async updateUser(req, res) {
    try {
      const { id } = req.params;
      const updateData = req.body;
      const user = await UserService.updateUser(id, updateData);

      res.json({
        success: true,
        data: user,
      });
    } catch (error) {
      if (error.name === "NotFoundError") {
        return res.status(404).json({
          success: false,
          error: "User not found",
        });
      }

      res.status(500).json({
        success: false,
        error: error.message,
      });
    }
  }

  async deleteUser(req, res) {
    try {
      const { id } = req.params;
      await UserService.deleteUser(id);

      res.status(204).send();
    } catch (error) {
      if (error.name === "NotFoundError") {
        return res.status(404).json({
          success: false,
          error: "User not found",
        });
      }

      res.status(500).json({
        success: false,
        error: error.message,
      });
    }
  }
}

const userController = new UserController();

// Routes
router.get("/", userController.listUsers);
router.get("/:id", userController.getUser);
router.post("/", userController.createUser);
router.put("/:id", userController.updateUser);
router.delete("/:id", userController.deleteUser);

module.exports = router;
```

### **HTTP Status Codes**

```javascript
// Proper HTTP status code usage
class StatusCodes {
  // Success responses
  static OK = 200;
  static CREATED = 201;
  static ACCEPTED = 202;
  static NO_CONTENT = 204;

  // Client error responses
  static BAD_REQUEST = 400;
  static UNAUTHORIZED = 401;
  static FORBIDDEN = 403;
  static NOT_FOUND = 404;
  static METHOD_NOT_ALLOWED = 405;
  static CONFLICT = 409;
  static UNPROCESSABLE_ENTITY = 422;
  static TOO_MANY_REQUESTS = 429;

  // Server error responses
  static INTERNAL_SERVER_ERROR = 500;
  static NOT_IMPLEMENTED = 501;
  static BAD_GATEWAY = 502;
  static SERVICE_UNAVAILABLE = 503;
  static GATEWAY_TIMEOUT = 504;
}

// Response helper
class ResponseHelper {
  static success(res, data, statusCode = StatusCodes.OK) {
    return res.status(statusCode).json({
      success: true,
      data,
      timestamp: new Date().toISOString(),
    });
  }

  static error(res, error, statusCode = StatusCodes.INTERNAL_SERVER_ERROR) {
    return res.status(statusCode).json({
      success: false,
      error: error.message || "Internal server error",
      timestamp: new Date().toISOString(),
      ...(process.env.NODE_ENV === "development" && { stack: error.stack }),
    });
  }

  static validationError(res, errors) {
    return res.status(StatusCodes.UNPROCESSABLE_ENTITY).json({
      success: false,
      error: "Validation failed",
      details: errors,
      timestamp: new Date().toISOString(),
    });
  }
}
```

## 🔍 GraphQL API Design

### **GraphQL Schema Design**

```javascript
const {
  GraphQLSchema,
  GraphQLObjectType,
  GraphQLString,
  GraphQLInt,
  GraphQLList,
  GraphQLNonNull,
} = require("graphql");

// User Type
const UserType = new GraphQLObjectType({
  name: "User",
  fields: () => ({
    id: { type: GraphQLNonNull(GraphQLString) },
    email: { type: GraphQLNonNull(GraphQLString) },
    name: { type: GraphQLString },
    posts: {
      type: GraphQLList(PostType),
      resolve: async (user, args, context) => {
        return await PostService.getPostsByUserId(user.id);
      },
    },
    createdAt: { type: GraphQLString },
    updatedAt: { type: GraphQLString },
  }),
});

// Post Type
const PostType = new GraphQLObjectType({
  name: "Post",
  fields: () => ({
    id: { type: GraphQLNonNull(GraphQLString) },
    title: { type: GraphQLNonNull(GraphQLString) },
    content: { type: GraphQLString },
    author: {
      type: UserType,
      resolve: async (post, args, context) => {
        return await UserService.getUserById(post.authorId);
      },
    },
    comments: {
      type: GraphQLList(CommentType),
      resolve: async (post, args, context) => {
        return await CommentService.getCommentsByPostId(post.id);
      },
    },
    createdAt: { type: GraphQLString },
    updatedAt: { type: GraphQLString },
  }),
});

// Comment Type
const CommentType = new GraphQLObjectType({
  name: "Comment",
  fields: () => ({
    id: { type: GraphQLNonNull(GraphQLString) },
    content: { type: GraphQLNonNull(GraphQLString) },
    author: {
      type: UserType,
      resolve: async (comment, args, context) => {
        return await UserService.getUserById(comment.authorId);
      },
    },
    post: {
      type: PostType,
      resolve: async (comment, args, context) => {
        return await PostService.getPostById(comment.postId);
      },
    },
    createdAt: { type: GraphQLString },
  }),
});

// Input Types
const UserInputType = new GraphQLInputObjectType({
  name: "UserInput",
  fields: {
    email: { type: GraphQLNonNull(GraphQLString) },
    name: { type: GraphQLString },
    password: { type: GraphQLNonNull(GraphQLString) },
  },
});

const PostInputType = new GraphQLInputObjectType({
  name: "PostInput",
  fields: {
    title: { type: GraphQLNonNull(GraphQLString) },
    content: { type: GraphQLString },
  },
});
```

### **GraphQL Resolvers**

```javascript
// Query resolvers
const QueryType = new GraphQLObjectType({
  name: "Query",
  fields: {
    user: {
      type: UserType,
      args: {
        id: { type: GraphQLNonNull(GraphQLString) },
      },
      resolve: async (parent, args, context) => {
        return await UserService.getUserById(args.id);
      },
    },
    users: {
      type: GraphQLList(UserType),
      args: {
        limit: { type: GraphQLInt, defaultValue: 10 },
        offset: { type: GraphQLInt, defaultValue: 0 },
      },
      resolve: async (parent, args, context) => {
        return await UserService.getUsers(args);
      },
    },
    post: {
      type: PostType,
      args: {
        id: { type: GraphQLNonNull(GraphQLString) },
      },
      resolve: async (parent, args, context) => {
        return await PostService.getPostById(args.id);
      },
    },
    posts: {
      type: GraphQLList(PostType),
      args: {
        limit: { type: GraphQLInt, defaultValue: 10 },
        offset: { type: GraphQLInt, defaultValue: 0 },
        authorId: { type: GraphQLString },
      },
      resolve: async (parent, args, context) => {
        return await PostService.getPosts(args);
      },
    },
  },
});

// Mutation resolvers
const MutationType = new GraphQLObjectType({
  name: "Mutation",
  fields: {
    createUser: {
      type: UserType,
      args: {
        input: { type: GraphQLNonNull(UserInputType) },
      },
      resolve: async (parent, args, context) => {
        return await UserService.createUser(args.input);
      },
    },
    updateUser: {
      type: UserType,
      args: {
        id: { type: GraphQLNonNull(GraphQLString) },
        input: { type: GraphQLNonNull(UserInputType) },
      },
      resolve: async (parent, args, context) => {
        return await UserService.updateUser(args.id, args.input);
      },
    },
    deleteUser: {
      type: GraphQLBoolean,
      args: {
        id: { type: GraphQLNonNull(GraphQLString) },
      },
      resolve: async (parent, args, context) => {
        await UserService.deleteUser(args.id);
        return true;
      },
    },
    createPost: {
      type: PostType,
      args: {
        input: { type: GraphQLNonNull(PostInputType) },
      },
      resolve: async (parent, args, context) => {
        // Add authorId from context (authenticated user)
        const postData = {
          ...args.input,
          authorId: context.user.id,
        };
        return await PostService.createPost(postData);
      },
    },
  },
});

// Schema
const schema = new GraphQLSchema({
  query: QueryType,
  mutation: MutationType,
});
```

### **GraphQL Server with Express**

```javascript
const express = require("express");
const { graphqlHTTP } = require("express-graphql");
const { buildSchema } = require("graphql");

const app = express();

// GraphQL endpoint
app.use(
  "/graphql",
  graphqlHTTP({
    schema: schema,
    rootValue: root,
    graphiql: process.env.NODE_ENV === "development",
    context: async ({ req }) => {
      // Authentication context
      const token = req.headers.authorization?.replace("Bearer ", "");
      const user = token ? await AuthService.verifyToken(token) : null;

      return {
        user,
        req,
      };
    },
    customFormatErrorFn: (error) => {
      console.error("GraphQL Error:", error);

      return {
        message: error.message,
        locations: error.locations,
        path: error.path,
        ...(process.env.NODE_ENV === "development" && { stack: error.stack }),
      };
    },
  })
);

// GraphQL Playground
if (process.env.NODE_ENV === "development") {
  app.use(
    "/playground",
    graphqlHTTP({
      schema: schema,
      graphiql: true,
    })
  );
}
```

## 🔧 gRPC API Design

### **Protocol Buffer Definitions**

```protobuf
// user.proto
syntax = "proto3";

package user;

service UserService {
    rpc GetUser(GetUserRequest) returns (GetUserResponse);
    rpc ListUsers(ListUsersRequest) returns (ListUsersResponse);
    rpc CreateUser(CreateUserRequest) returns (CreateUserResponse);
    rpc UpdateUser(UpdateUserRequest) returns (UpdateUserResponse);
    rpc DeleteUser(DeleteUserRequest) returns (DeleteUserResponse);
    rpc StreamUsers(StreamUsersRequest) returns (stream User);
}

message User {
    string id = 1;
    string email = 2;
    string name = 3;
    int64 created_at = 4;
    int64 updated_at = 5;
}

message GetUserRequest {
    string id = 1;
}

message GetUserResponse {
    User user = 1;
}

message ListUsersRequest {
    int32 limit = 1;
    int32 offset = 2;
    string sort_by = 3;
    string sort_order = 4;
}

message ListUsersResponse {
    repeated User users = 1;
    int32 total = 2;
}

message CreateUserRequest {
    string email = 1;
    string name = 2;
    string password = 3;
}

message CreateUserResponse {
    User user = 1;
}

message UpdateUserRequest {
    string id = 1;
    string email = 2;
    string name = 3;
}

message UpdateUserResponse {
    User user = 1;
}

message DeleteUserRequest {
    string id = 1;
}

message DeleteUserResponse {
    bool success = 1;
}

message StreamUsersRequest {
    int32 batch_size = 1;
}
```

### **gRPC Server Implementation**

```javascript
const grpc = require("@grpc/grpc-js");
const protoLoader = require("@grpc/proto-loader");
const path = require("path");

// Load proto file
const PROTO_PATH = path.join(__dirname, "user.proto");
const packageDefinition = protoLoader.loadSync(PROTO_PATH, {
  keepCase: true,
  longs: String,
  enums: String,
  defaults: true,
  oneofs: true,
});

const userProto = grpc.loadPackageDefinition(packageDefinition).user;

// gRPC Service Implementation
class UserService {
  async getUser(call, callback) {
    try {
      const { id } = call.request;
      const user = await UserService.getUserById(id);

      if (!user) {
        return callback({
          code: grpc.status.NOT_FOUND,
          message: "User not found",
        });
      }

      callback(null, { user });
    } catch (error) {
      callback({
        code: grpc.status.INTERNAL,
        message: error.message,
      });
    }
  }

  async listUsers(call, callback) {
    try {
      const { limit, offset, sort_by, sort_order } = call.request;
      const result = await UserService.getUsers({
        limit,
        offset,
        sort: sort_by,
        order: sort_order,
      });

      callback(null, {
        users: result.users,
        total: result.total,
      });
    } catch (error) {
      callback({
        code: grpc.status.INTERNAL,
        message: error.message,
      });
    }
  }

  async createUser(call, callback) {
    try {
      const userData = call.request;
      const user = await UserService.createUser(userData);

      callback(null, { user });
    } catch (error) {
      if (error.name === "ValidationError") {
        return callback({
          code: grpc.status.INVALID_ARGUMENT,
          message: error.message,
        });
      }

      callback({
        code: grpc.status.INTERNAL,
        message: error.message,
      });
    }
  }

  async updateUser(call, callback) {
    try {
      const { id, ...updateData } = call.request;
      const user = await UserService.updateUser(id, updateData);

      callback(null, { user });
    } catch (error) {
      if (error.name === "NotFoundError") {
        return callback({
          code: grpc.status.NOT_FOUND,
          message: "User not found",
        });
      }

      callback({
        code: grpc.status.INTERNAL,
        message: error.message,
      });
    }
  }

  async deleteUser(call, callback) {
    try {
      const { id } = call.request;
      await UserService.deleteUser(id);

      callback(null, { success: true });
    } catch (error) {
      if (error.name === "NotFoundError") {
        return callback({
          code: grpc.status.NOT_FOUND,
          message: "User not found",
        });
      }

      callback({
        code: grpc.status.INTERNAL,
        message: error.message,
      });
    }
  }

  // Streaming example
  async streamUsers(call) {
    try {
      const { batch_size = 10 } = call.request;
      let offset = 0;

      while (true) {
        const users = await UserService.getUsers({
          limit: batch_size,
          offset,
        });

        if (users.length === 0) {
          break;
        }

        for (const user of users) {
          call.write(user);
        }

        offset += batch_size;

        // Add delay to prevent overwhelming
        await new Promise((resolve) => setTimeout(resolve, 100));
      }

      call.end();
    } catch (error) {
      call.emit("error", {
        code: grpc.status.INTERNAL,
        message: error.message,
      });
    }
  }
}

// Create gRPC server
const server = new grpc.Server();
server.addService(userProto.UserService.service, new UserService());

const PORT = process.env.GRPC_PORT || 50051;
server.bindAsync(
  `0.0.0.0:${PORT}`,
  grpc.ServerCredentials.createInsecure(),
  (err, port) => {
    if (err) {
      console.error("Failed to start gRPC server:", err);
      return;
    }

    console.log(`gRPC server running on port ${port}`);
    server.start();
  }
);
```

## 🔒 API Security

### **Authentication & Authorization**

```javascript
const jwt = require("jsonwebtoken");
const bcrypt = require("bcrypt");
const rateLimit = require("express-rate-limit");

// JWT Authentication
class AuthService {
  static generateToken(user) {
    return jwt.sign(
      {
        id: user.id,
        email: user.email,
        role: user.role,
      },
      process.env.JWT_SECRET,
      { expiresIn: "24h" }
    );
  }

  static verifyToken(token) {
    try {
      return jwt.verify(token, process.env.JWT_SECRET);
    } catch (error) {
      throw new Error("Invalid token");
    }
  }

  static async hashPassword(password) {
    const saltRounds = 12;
    return await bcrypt.hash(password, saltRounds);
  }

  static async comparePassword(password, hash) {
    return await bcrypt.compare(password, hash);
  }
}

// Authentication middleware
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers["authorization"];
  const token = authHeader && authHeader.split(" ")[1];

  if (!token) {
    return res.status(401).json({
      success: false,
      error: "Access token required",
    });
  }

  try {
    const user = AuthService.verifyToken(token);
    req.user = user;
    next();
  } catch (error) {
    return res.status(403).json({
      success: false,
      error: "Invalid or expired token",
    });
  }
};

// Authorization middleware
const authorize = (roles) => {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({
        success: false,
        error: "Authentication required",
      });
    }

    if (!roles.includes(req.user.role)) {
      return res.status(403).json({
        success: false,
        error: "Insufficient permissions",
      });
    }

    next();
  };
};

// Rate limiting
const createRateLimit = (windowMs, max, message) => {
  return rateLimit({
    windowMs,
    max,
    message: {
      success: false,
      error: message,
    },
    standardHeaders: true,
    legacyHeaders: false,
  });
};

// Apply rate limits
const authLimiter = createRateLimit(
  15 * 60 * 1000, // 15 minutes
  5, // 5 attempts
  "Too many authentication attempts, please try again later"
);

const apiLimiter = createRateLimit(
  15 * 60 * 1000, // 15 minutes
  100, // 100 requests
  "Too many requests, please try again later"
);

// Apply middleware
app.use("/api/v1/auth", authLimiter);
app.use("/api/v1", apiLimiter);
```

### **Input Validation**

```javascript
const Joi = require("joi");

// Validation schemas
const schemas = {
  user: {
    create: Joi.object({
      email: Joi.string().email().required(),
      name: Joi.string().min(2).max(50).required(),
      password: Joi.string()
        .min(8)
        .pattern(
          /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/
        )
        .required(),
    }),
    update: Joi.object({
      email: Joi.string().email(),
      name: Joi.string().min(2).max(50),
    }),
  },
  post: {
    create: Joi.object({
      title: Joi.string().min(1).max(200).required(),
      content: Joi.string().min(1).max(10000).required(),
    }),
    update: Joi.object({
      title: Joi.string().min(1).max(200),
      content: Joi.string().min(1).max(10000),
    }),
  },
};

// Validation middleware
const validate = (schema) => {
  return (req, res, next) => {
    const { error, value } = schema.validate(req.body, {
      abortEarly: false,
      stripUnknown: true,
    });

    if (error) {
      const details = error.details.map((detail) => ({
        field: detail.path.join("."),
        message: detail.message,
      }));

      return res.status(422).json({
        success: false,
        error: "Validation failed",
        details,
      });
    }

    req.body = value;
    next();
  };
};

// Usage
router.post("/users", validate(schemas.user.create), userController.createUser);
router.put(
  "/users/:id",
  validate(schemas.user.update),
  userController.updateUser
);
```

## 📚 API Documentation

### **OpenAPI/Swagger Documentation**

```javascript
const swaggerJsdoc = require("swagger-jsdoc");
const swaggerUi = require("swagger-ui-express");

// Swagger configuration
const swaggerOptions = {
  definition: {
    openapi: "3.0.0",
    info: {
      title: "User API",
      version: "1.0.0",
      description: "A comprehensive user management API",
      contact: {
        name: "API Support",
        email: "support@example.com",
      },
    },
    servers: [
      {
        url: "http://localhost:3000/api/v1",
        description: "Development server",
      },
    ],
    components: {
      securitySchemes: {
        bearerAuth: {
          type: "http",
          scheme: "bearer",
          bearerFormat: "JWT",
        },
      },
      schemas: {
        User: {
          type: "object",
          required: ["id", "email", "name"],
          properties: {
            id: {
              type: "string",
              description: "User unique identifier",
            },
            email: {
              type: "string",
              format: "email",
              description: "User email address",
            },
            name: {
              type: "string",
              description: "User full name",
            },
            createdAt: {
              type: "string",
              format: "date-time",
              description: "User creation timestamp",
            },
            updatedAt: {
              type: "string",
              format: "date-time",
              description: "User last update timestamp",
            },
          },
        },
        Error: {
          type: "object",
          properties: {
            success: {
              type: "boolean",
              example: false,
            },
            error: {
              type: "string",
              description: "Error message",
            },
            timestamp: {
              type: "string",
              format: "date-time",
            },
          },
        },
      },
    },
    security: [
      {
        bearerAuth: [],
      },
    ],
  },
  apis: ["./routes/*.js"],
};

const specs = swaggerJsdoc(swaggerOptions);

// Swagger UI
app.use(
  "/api-docs",
  swaggerUi.serve,
  swaggerUi.setup(specs, {
    explorer: true,
    customCss: ".swagger-ui .topbar { display: none }",
    customSiteTitle: "User API Documentation",
  })
);

// JSDoc comments for Swagger
/**
 * @swagger
 * /users:
 *   get:
 *     summary: Get all users
 *     tags: [Users]
 *     parameters:
 *       - in: query
 *         name: page
 *         schema:
 *           type: integer
 *           default: 1
 *         description: Page number
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           default: 10
 *         description: Number of users per page
 *     responses:
 *       200:
 *         description: List of users
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 data:
 *                   type: array
 *                   items:
 *                     $ref: '#/components/schemas/User'
 *       500:
 *         description: Internal server error
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 */
```

## 🧪 API Testing

### **Unit Testing with Jest**

```javascript
const request = require("supertest");
const app = require("../app");
const UserService = require("../services/UserService");

describe("User API", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("GET /api/v1/users", () => {
    it("should return list of users", async () => {
      const mockUsers = [
        { id: "1", email: "user1@example.com", name: "User 1" },
        { id: "2", email: "user2@example.com", name: "User 2" },
      ];

      jest.spyOn(UserService, "getUsers").mockResolvedValue({
        users: mockUsers,
        total: 2,
      });

      const response = await request(app).get("/api/v1/users").expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.data).toHaveLength(2);
      expect(response.body.data[0]).toHaveProperty("id");
      expect(response.body.data[0]).toHaveProperty("email");
    });

    it("should handle pagination", async () => {
      jest.spyOn(UserService, "getUsers").mockResolvedValue({
        users: [],
        total: 0,
      });

      const response = await request(app)
        .get("/api/v1/users?page=2&limit=5")
        .expect(200);

      expect(UserService.getUsers).toHaveBeenCalledWith({
        page: 2,
        limit: 5,
        sort: "createdAt",
        order: "desc",
      });
    });
  });

  describe("POST /api/v1/users", () => {
    it("should create a new user", async () => {
      const userData = {
        email: "newuser@example.com",
        name: "New User",
        password: "SecurePass123!",
      };

      const mockUser = {
        id: "3",
        ...userData,
        createdAt: new Date().toISOString(),
      };

      jest.spyOn(UserService, "createUser").mockResolvedValue(mockUser);

      const response = await request(app)
        .post("/api/v1/users")
        .send(userData)
        .expect(201);

      expect(response.body.success).toBe(true);
      expect(response.body.data.email).toBe(userData.email);
      expect(response.body.data).not.toHaveProperty("password");
    });

    it("should validate input data", async () => {
      const invalidData = {
        email: "invalid-email",
        name: "",
        password: "weak",
      };

      const response = await request(app)
        .post("/api/v1/users")
        .send(invalidData)
        .expect(422);

      expect(response.body.success).toBe(false);
      expect(response.body.details).toBeDefined();
    });
  });

  describe("Authentication", () => {
    it("should require authentication for protected routes", async () => {
      await request(app).get("/api/v1/users/1").expect(401);
    });

    it("should accept valid JWT token", async () => {
      const token = AuthService.generateToken({
        id: "1",
        email: "user@example.com",
        role: "user",
      });

      jest.spyOn(UserService, "getUserById").mockResolvedValue({
        id: "1",
        email: "user@example.com",
        name: "Test User",
      });

      await request(app)
        .get("/api/v1/users/1")
        .set("Authorization", `Bearer ${token}`)
        .expect(200);
    });
  });
});
```

### **Integration Testing**

```javascript
describe("API Integration Tests", () => {
  let server;
  let authToken;

  beforeAll(async () => {
    server = app.listen(0);

    // Create test user and get auth token
    const userData = {
      email: "test@example.com",
      name: "Test User",
      password: "TestPass123!",
    };

    const user = await UserService.createUser(userData);
    authToken = AuthService.generateToken(user);
  });

  afterAll(async () => {
    await server.close();
  });

  describe("Complete User Workflow", () => {
    it("should handle complete CRUD operations", async () => {
      // Create user
      const createResponse = await request(app)
        .post("/api/v1/users")
        .send({
          email: "workflow@example.com",
          name: "Workflow User",
          password: "WorkflowPass123!",
        })
        .expect(201);

      const userId = createResponse.body.data.id;

      // Get user
      await request(app)
        .get(`/api/v1/users/${userId}`)
        .set("Authorization", `Bearer ${authToken}`)
        .expect(200);

      // Update user
      await request(app)
        .put(`/api/v1/users/${userId}`)
        .set("Authorization", `Bearer ${authToken}`)
        .send({
          name: "Updated Workflow User",
        })
        .expect(200);

      // Delete user
      await request(app)
        .delete(`/api/v1/users/${userId}`)
        .set("Authorization", `Bearer ${authToken}`)
        .expect(204);
    });
  });
});
```

## ⚡ API Performance

### **Caching Strategy**

```javascript
const Redis = require("redis");
const client = Redis.createClient({
  host: process.env.REDIS_HOST || "localhost",
  port: process.env.REDIS_PORT || 6379,
});

class CacheService {
  static async get(key) {
    try {
      const value = await client.get(key);
      return value ? JSON.parse(value) : null;
    } catch (error) {
      console.error("Cache get error:", error);
      return null;
    }
  }

  static async set(key, value, ttl = 3600) {
    try {
      await client.setex(key, ttl, JSON.stringify(value));
    } catch (error) {
      console.error("Cache set error:", error);
    }
  }

  static async del(key) {
    try {
      await client.del(key);
    } catch (error) {
      console.error("Cache delete error:", error);
    }
  }

  static async invalidatePattern(pattern) {
    try {
      const keys = await client.keys(pattern);
      if (keys.length > 0) {
        await client.del(...keys);
      }
    } catch (error) {
      console.error("Cache invalidation error:", error);
    }
  }
}

// Caching middleware
const cache = (ttl = 3600) => {
  return async (req, res, next) => {
    const key = `cache:${req.originalUrl}`;

    try {
      const cached = await CacheService.get(key);
      if (cached) {
        return res.json(cached);
      }

      // Store original res.json
      const originalJson = res.json;
      res.json = function (data) {
        // Cache the response
        CacheService.set(key, data, ttl);
        originalJson.call(this, data);
      };

      next();
    } catch (error) {
      console.error("Cache middleware error:", error);
      next();
    }
  };
};

// Usage
router.get("/users", cache(300), userController.listUsers); // 5 minutes cache
router.get("/users/:id", cache(600), userController.getUser); // 10 minutes cache
```

### **Database Optimization**

```javascript
// Database connection pooling
const { Pool } = require("pg");

const pool = new Pool({
  host: process.env.DB_HOST,
  port: process.env.DB_PORT,
  database: process.env.DB_NAME,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  max: 20, // Maximum number of clients in the pool
  idleTimeoutMillis: 30000, // Close idle clients after 30 seconds
  connectionTimeoutMillis: 2000, // Return an error after 2 seconds if connection could not be established
});

// Query optimization
class DatabaseService {
  static async getUsersWithPagination(options) {
    const {
      page = 1,
      limit = 10,
      sort = "created_at",
      order = "DESC",
    } = options;
    const offset = (page - 1) * limit;

    // Use prepared statements for better performance
    const query = `
            SELECT id, email, name, created_at, updated_at
            FROM users
            ORDER BY ${sort} ${order}
            LIMIT $1 OFFSET $2
        `;

    const countQuery = "SELECT COUNT(*) FROM users";

    const [usersResult, countResult] = await Promise.all([
      pool.query(query, [limit, offset]),
      pool.query(countQuery),
    ]);

    return {
      users: usersResult.rows,
      total: parseInt(countResult.rows[0].count),
    };
  }

  static async getUserById(id) {
    const query = `
            SELECT id, email, name, created_at, updated_at
            FROM users
            WHERE id = $1
        `;

    const result = await pool.query(query, [id]);
    return result.rows[0] || null;
  }
}
```

## 🔄 API Versioning

### **URL Versioning**

```javascript
// API versioning with Express
const v1Routes = require("./routes/v1");
const v2Routes = require("./routes/v2");

app.use("/api/v1", v1Routes);
app.use("/api/v2", v2Routes);

// Version-specific middleware
const versionMiddleware = (version) => {
  return (req, res, next) => {
    req.apiVersion = version;
    next();
  };
};

// Version-specific controllers
class UserControllerV1 {
  async getUser(req, res) {
    // V1 implementation
    const user = await UserService.getUserById(req.params.id);
    res.json({
      success: true,
      data: user,
    });
  }
}

class UserControllerV2 {
  async getUser(req, res) {
    // V2 implementation with additional fields
    const user = await UserService.getUserById(req.params.id);
    const userWithStats = await UserService.getUserWithStats(user.id);

    res.json({
      success: true,
      data: userWithStats,
      version: "2.0",
    });
  }
}
```

### **Header Versioning**

```javascript
// Header-based versioning
const versionMiddleware = (req, res, next) => {
  const apiVersion = req.headers["api-version"] || "1.0";
  req.apiVersion = apiVersion;

  // Route to appropriate controller based on version
  if (apiVersion.startsWith("2.")) {
    req.controller = UserControllerV2;
  } else {
    req.controller = UserControllerV1;
  }

  next();
};

// Usage
app.use("/api/users", versionMiddleware, (req, res, next) => {
  const controller = new req.controller();
  return controller[req.method.toLowerCase()](req, res, next/);
});
```

---

**🎉 Master these API design patterns to build robust, scalable, and maintainable Node.js APIs!**

**Good luck with your API development journey! 🚀**
