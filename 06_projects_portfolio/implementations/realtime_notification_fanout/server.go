package main

import (
	"encoding/json"
	"log"
	"net/http"
)

var dispatcher = NewDispatcher()

type dispatchRequest struct {
	UserID  string `json:"userId"`
	Channel string `json:"channel"`
	Content string `json:"content"`
}

type dispatchResponse struct {
	Status string `json:"status"`
}

func dispatchHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	var dr dispatchRequest
	if err := json.NewDecoder(r.Body).Decode(&dr); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		return
	}
	err := dispatcher.Dispatch(Notification{UserID: dr.UserID, Channel: dr.Channel, Content: dr.Content})
	w.Header().Set("Content-Type", "application/json")
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}
	_ = json.NewEncoder(w).Encode(dispatchResponse{Status: "ok"})
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/notify/dispatch", dispatchHandler)
	log.Println("notification fan-out server on :8084")
	log.Fatal(http.ListenAndServe(":8084", mux))
}
