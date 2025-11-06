package main

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"
)

var cfgService = NewConfigService()

func getConfigHandler(w http.ResponseWriter, r *http.Request) {
	key := strings.TrimPrefix(r.URL.Path, "/config/")
	doc, ok := cfgService.Get(key)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"found": ok, "doc": doc})
}

func setConfigHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPut {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	key := strings.TrimPrefix(r.URL.Path, "/config/")
	val := r.URL.Query().Get("value")
	doc := cfgService.Set(key, val)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(doc)
}

func auditHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(cfgService.Versions())
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/config/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			getConfigHandler(w, r)
			return
		}
		if r.Method == http.MethodPut {
			setConfigHandler(w, r)
			return
		}
		w.WriteHeader(http.StatusMethodNotAllowed)
	})
	mux.HandleFunc("/config_audit", auditHandler)
	log.Println("config distribution server on :8083")
	log.Fatal(http.ListenAndServe(":8083", mux))
}
