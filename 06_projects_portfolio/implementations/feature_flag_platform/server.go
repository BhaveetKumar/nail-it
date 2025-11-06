package main

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"
)

var store = NewFlagStore()

func getFlagHandler(w http.ResponseWriter, r *http.Request) {
	name := strings.TrimPrefix(r.URL.Path, "/flags/")
	flag, ok := store.Get(name)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"found": ok, "flag": flag})
}

func setFlagHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPut {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	name := strings.TrimPrefix(r.URL.Path, "/flags/")
	on := r.URL.Query().Get("on") == "true"
	store.Set(Flag{Name: name, On: on})
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"updated": true, "version": store.version})
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/flags/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			getFlagHandler(w, r)
			return
		}
		if r.Method == http.MethodPut {
			setFlagHandler(w, r)
			return
		}
		w.WriteHeader(http.StatusMethodNotAllowed)
	})
	log.Println("feature flag server on :8082")
	log.Fatal(http.ListenAndServe(":8082", mux))
}
