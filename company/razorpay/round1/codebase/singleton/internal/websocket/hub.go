package websocket

import (
	"encoding/json"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"singleton-service/internal/logger"
)

// Hub implements Singleton pattern for WebSocket hub
type Hub struct {
	clients    map[*Client]bool
	broadcast  chan []byte
	register   chan *Client
	unregister chan *Client
	mutex      sync.RWMutex
}

// Client represents a WebSocket client
type Client struct {
	hub      *Hub
	conn     *websocket.Conn
	send     chan []byte
	userID   string
	clientID string
}

// Message represents a WebSocket message
type Message struct {
	Type      string      `json:"type"`
	Data      interface{} `json:"data"`
	Timestamp int64       `json:"timestamp"`
	UserID    string      `json:"user_id,omitempty"`
	ClientID  string      `json:"client_id,omitempty"`
}

var (
	wsHub *Hub
	hubOnce sync.Once
)

// GetWebSocketHub returns the singleton instance of WebSocket hub
func GetWebSocketHub() *Hub {
	hubOnce.Do(func() {
		wsHub = &Hub{
			clients:    make(map[*Client]bool),
			broadcast:  make(chan []byte),
			register:   make(chan *Client),
			unregister: make(chan *Client),
		}
	})
	return wsHub
}

// Run starts the WebSocket hub
func (h *Hub) Run() {
	log := logger.GetLogger()
	log.Info("Starting WebSocket hub")

	for {
		select {
		case client := <-h.register:
			h.mutex.Lock()
			h.clients[client] = true
			h.mutex.Unlock()
			log.Info("Client registered", "client_id", client.clientID, "user_id", client.userID)

		case client := <-h.unregister:
			h.mutex.Lock()
			if _, ok := h.clients[client]; ok {
				delete(h.clients, client)
				close(client.send)
			}
			h.mutex.Unlock()
			log.Info("Client unregistered", "client_id", client.clientID, "user_id", client.userID)

		case message := <-h.broadcast:
			h.mutex.RLock()
			for client := range h.clients {
				select {
				case client.send <- message:
				default:
					close(client.send)
					delete(h.clients, client)
				}
			}
			h.mutex.RUnlock()
		}
	}
}

// RegisterClient registers a new WebSocket client
func (h *Hub) RegisterClient(conn *websocket.Conn, userID, clientID string) *Client {
	client := &Client{
		hub:      h,
		conn:     conn,
		send:     make(chan []byte, 256),
		userID:   userID,
		clientID: clientID,
	}

	h.register <- client
	return client
}

// UnregisterClient unregisters a WebSocket client
func (h *Hub) UnregisterClient(client *Client) {
	h.unregister <- client
}

// BroadcastMessage broadcasts a message to all connected clients
func (h *Hub) BroadcastMessage(message *Message) error {
	messageBytes, err := json.Marshal(message)
	if err != nil {
		return err
	}

	h.broadcast <- messageBytes
	return nil
}

// SendToUser sends a message to a specific user
func (h *Hub) SendToUser(userID string, message *Message) error {
	messageBytes, err := json.Marshal(message)
	if err != nil {
		return err
	}

	h.mutex.RLock()
	defer h.mutex.RUnlock()

	for client := range h.clients {
		if client.userID == userID {
			select {
			case client.send <- messageBytes:
			default:
				close(client.send)
				delete(h.clients, client)
			}
		}
	}

	return nil
}

// SendToClient sends a message to a specific client
func (h *Hub) SendToClient(clientID string, message *Message) error {
	messageBytes, err := json.Marshal(message)
	if err != nil {
		return err
	}

	h.mutex.RLock()
	defer h.mutex.RUnlock()

	for client := range h.clients {
		if client.clientID == clientID {
			select {
			case client.send <- messageBytes:
			default:
				close(client.send)
				delete(h.clients, client)
			}
			break
		}
	}

	return nil
}

// GetConnectedClients returns the number of connected clients
func (h *Hub) GetConnectedClients() int {
	h.mutex.RLock()
	defer h.mutex.RUnlock()
	return len(h.clients)
}

// GetConnectedUsers returns the number of unique connected users
func (h *Hub) GetConnectedUsers() int {
	h.mutex.RLock()
	defer h.mutex.RUnlock()

	users := make(map[string]bool)
	for client := range h.clients {
		users[client.userID] = true
	}
	return len(users)
}

// ReadPump handles reading messages from the WebSocket connection
func (c *Client) ReadPump() {
	defer func() {
		c.hub.UnregisterClient(c)
		c.conn.Close()
	}()

	c.conn.SetReadLimit(512)
	c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	c.conn.SetPongHandler(func(string) error {
		c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})

	for {
		_, _, err := c.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log := logger.GetLogger()
				log.Error("WebSocket error", "error", err)
			}
			break
		}
	}
}

// WritePump handles writing messages to the WebSocket connection
func (c *Client) WritePump() {
	ticker := time.NewTicker(54 * time.Second)
	defer func() {
		ticker.Stop()
		c.conn.Close()
	}()

	for {
		select {
		case message, ok := <-c.send:
			c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if !ok {
				c.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}

			w, err := c.conn.NextWriter(websocket.TextMessage)
			if err != nil {
				return
			}
			w.Write(message)

			// Add queued chat messages to the current websocket message
			n := len(c.send)
			for i := 0; i < n; i++ {
				w.Write([]byte{'\n'})
				w.Write(<-c.send)
			}

			if err := w.Close(); err != nil {
				return
			}

		case <-ticker.C:
			c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

// SendMessage sends a message to the client
func (c *Client) SendMessage(message *Message) error {
	messageBytes, err := json.Marshal(message)
	if err != nil {
		return err
	}

	select {
	case c.send <- messageBytes:
		return nil
	default:
		return websocket.ErrCloseSent
	}
}

// GetUserID returns the user ID of the client
func (c *Client) GetUserID() string {
	return c.userID
}

// GetClientID returns the client ID of the client
func (c *Client) GetClientID() string {
	return c.clientID
}
