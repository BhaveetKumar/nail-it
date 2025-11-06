package main

import (
	"sync/atomic"
)

type Flag struct {
	Name string
	On   bool
}

type FlagStore struct {
	flags   atomic.Value // map[string]Flag
	version uint64
}

func NewFlagStore() *FlagStore {
	fs := &FlagStore{}
	fs.flags.Store(map[string]Flag{"new_ui": {Name: "new_ui", On: true}})
	return fs
}

func (fs *FlagStore) Get(name string) (Flag, bool) {
	m := fs.flags.Load().(map[string]Flag)
	f, ok := m[name]
	return f, ok
}

func (fs *FlagStore) Set(f Flag) {
	m := fs.flags.Load().(map[string]Flag)
	newMap := make(map[string]Flag, len(m)+1)
	for k, v := range m {
		newMap[k] = v
	}
	newMap[f.Name] = f
	fs.flags.Store(newMap)
	fs.version++
}

// DemoFlags performs a simple retrieval and mutation returning counts.
func DemoFlags() (initial bool, after bool, version uint64) {
	store := NewFlagStore()
	f, _ := store.Get("new_ui")
	store.Set(Flag{Name: "beta_payment", On: false})
	f2, _ := store.Get("beta_payment")
	return f.On, f2.On, store.version
}
