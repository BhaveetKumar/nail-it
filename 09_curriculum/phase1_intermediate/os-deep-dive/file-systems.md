# File Systems

## Overview

This module covers file system implementation concepts including inode structures, directory management, file operations, caching strategies, and file system optimization. These concepts are essential for understanding how operating systems manage persistent storage.

## Table of Contents

1. [Inode Structure](#inode-structure)
2. [Directory Management](#directory-management)
3. [File Operations](#file-operations)
4. [Caching Strategies](#caching-strategies)
5. [File System Optimization](#file-system-optimization)
6. [Applications](#applications)
7. [Complexity Analysis](#complexity-analysis)
8. [Follow-up Questions](#follow-up-questions)

## Inode Structure

### Theory

An inode (index node) is a data structure that stores metadata about a file, including file size, permissions, timestamps, and pointers to data blocks. It's the fundamental building block of Unix-like file systems.

### Inode Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "time"
)

type Inode struct {
    ID          int
    Mode        uint32
    UID         int
    GID         int
    Size        int64
    Blocks      int64
    Atime       time.Time
    Mtime       time.Time
    Ctime       time.Time
    DirectBlocks [12]int64  // Direct block pointers
    IndirectBlock int64     // Single indirect block pointer
    DoubleIndirectBlock int64 // Double indirect block pointer
    TripleIndirectBlock int64 // Triple indirect block pointer
    LinkCount   int
    Type        string
}

type FileSystem struct {
    inodes      map[int]*Inode
    dataBlocks  map[int64][]byte
    blockSize   int
    nextInodeID int
    nextBlockID int64
}

func NewFileSystem(blockSize int) *FileSystem {
    return &FileSystem{
        inodes:      make(map[int]*Inode),
        dataBlocks:  make(map[int64][]byte),
        blockSize:   blockSize,
        nextInodeID: 1,
        nextBlockID: 1,
    }
}

func (fs *FileSystem) CreateInode(mode uint32, uid, gid int) *Inode {
    inode := &Inode{
        ID:          fs.nextInodeID,
        Mode:        mode,
        UID:         uid,
        GID:         gid,
        Size:        0,
        Blocks:      0,
        Atime:       time.Now(),
        Mtime:       time.Now(),
        Ctime:       time.Now(),
        LinkCount:   1,
        Type:        "file",
    }
    
    fs.inodes[fs.nextInodeID] = inode
    fs.nextInodeID++
    
    fmt.Printf("Created inode %d\n", inode.ID)
    return inode
}

func (fs *FileSystem) GetInode(id int) (*Inode, bool) {
    inode, exists := fs.inodes[id]
    if exists {
        inode.Atime = time.Now()
    }
    return inode, exists
}

func (fs *FileSystem) UpdateInode(id int, size int64) {
    if inode, exists := fs.inodes[id]; exists {
        inode.Size = size
        inode.Mtime = time.Now()
        inode.Ctime = time.Now()
        
        // Calculate number of blocks needed
        blocksNeeded := (size + int64(fs.blockSize) - 1) / int64(fs.blockSize)
        inode.Blocks = blocksNeeded
        
        fmt.Printf("Updated inode %d: size=%d, blocks=%d\n", id, size, blocksNeeded)
    }
}

func (fs *FileSystem) AllocateBlock() int64 {
    blockID := fs.nextBlockID
    fs.dataBlocks[blockID] = make([]byte, fs.blockSize)
    fs.nextBlockID++
    return blockID
}

func (fs *FileSystem) WriteToBlock(blockID int64, data []byte, offset int) {
    if block, exists := fs.dataBlocks[blockID]; exists {
        copy(block[offset:], data)
        fmt.Printf("Wrote %d bytes to block %d at offset %d\n", len(data), blockID, offset)
    }
}

func (fs *FileSystem) ReadFromBlock(blockID int64, offset, length int) []byte {
    if block, exists := fs.dataBlocks[blockID]; exists {
        end := offset + length
        if end > len(block) {
            end = len(block)
        }
        return block[offset:end]
    }
    return nil
}

func (fs *FileSystem) PrintInode(id int) {
    if inode, exists := fs.inodes[id]; exists {
        fmt.Printf("Inode %d:\n", id)
        fmt.Printf("  Mode: %o\n", inode.Mode)
        fmt.Printf("  UID: %d, GID: %d\n", inode.UID, inode.GID)
        fmt.Printf("  Size: %d bytes\n", inode.Size)
        fmt.Printf("  Blocks: %d\n", inode.Blocks)
        fmt.Printf("  Atime: %s\n", inode.Atime.Format(time.RFC3339))
        fmt.Printf("  Mtime: %s\n", inode.Mtime.Format(time.RFC3339))
        fmt.Printf("  Ctime: %s\n", inode.Ctime.Format(time.RFC3339))
        fmt.Printf("  Link Count: %d\n", inode.LinkCount)
        fmt.Printf("  Type: %s\n", inode.Type)
        
        fmt.Printf("  Direct Blocks: ")
        for i, block := range inode.DirectBlocks {
            if block != 0 {
                fmt.Printf("%d ", block)
            } else {
                fmt.Printf("0 ")
            }
            if i == 11 {
                fmt.Println()
            }
        }
        
        if inode.IndirectBlock != 0 {
            fmt.Printf("  Indirect Block: %d\n", inode.IndirectBlock)
        }
        if inode.DoubleIndirectBlock != 0 {
            fmt.Printf("  Double Indirect Block: %d\n", inode.DoubleIndirectBlock)
        }
        if inode.TripleIndirectBlock != 0 {
            fmt.Printf("  Triple Indirect Block: %d\n", inode.TripleIndirectBlock)
        }
    } else {
        fmt.Printf("Inode %d not found\n", id)
    }
}

func main() {
    fs := NewFileSystem(4096) // 4KB blocks
    
    fmt.Println("File System Demo:")
    
    // Create a file inode
    inode := fs.CreateInode(0644, 1000, 1000)
    fs.UpdateInode(inode.ID, 8192) // 8KB file
    
    // Allocate some blocks
    block1 := fs.AllocateBlock()
    block2 := fs.AllocateBlock()
    
    // Update inode with block pointers
    inode.DirectBlocks[0] = block1
    inode.DirectBlocks[1] = block2
    
    // Write data to blocks
    data1 := []byte("Hello, World! This is block 1.")
    data2 := []byte("This is block 2 with more data.")
    
    fs.WriteToBlock(block1, data1, 0)
    fs.WriteToBlock(block2, data2, 0)
    
    // Print inode information
    fs.PrintInode(inode.ID)
    
    // Read data back
    fmt.Println("\nReading data:")
    readData1 := fs.ReadFromBlock(block1, 0, len(data1))
    readData2 := fs.ReadFromBlock(block2, 0, len(data2))
    
    fmt.Printf("Block 1: %s\n", string(readData1))
    fmt.Printf("Block 2: %s\n", string(readData2))
}
```

## Directory Management

### Theory

Directories are special files that contain entries mapping names to inode numbers. They provide the hierarchical structure of the file system.

### Directory Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "time"
)

type DirectoryEntry struct {
    Name    string
    InodeID int
    Type    string
}

type Directory struct {
    InodeID int
    Entries map[string]*DirectoryEntry
    Parent  *Directory
    Name    string
}

type FileSystemManager struct {
    fs        *FileSystem
    rootDir   *Directory
    currentDir *Directory
    dirs      map[int]*Directory
}

func NewFileSystemManager(fs *FileSystem) *FileSystemManager {
    fsm := &FileSystemManager{
        fs:   fs,
        dirs: make(map[int]*Directory),
    }
    
    // Create root directory
    rootInode := fs.CreateInode(0755, 0, 0)
    rootInode.Type = "directory"
    
    rootDir := &Directory{
        InodeID: rootInode.ID,
        Entries: make(map[string]*DirectoryEntry),
        Parent:  nil,
        Name:    "/",
    }
    
    fsm.rootDir = rootDir
    fsm.currentDir = rootDir
    fsm.dirs[rootInode.ID] = rootDir
    
    return fsm
}

func (fsm *FileSystemManager) CreateDirectory(name string) bool {
    if _, exists := fsm.currentDir.Entries[name]; exists {
        fmt.Printf("Directory '%s' already exists\n", name)
        return false
    }
    
    // Create inode for directory
    dirInode := fsm.fs.CreateInode(0755, 1000, 1000)
    dirInode.Type = "directory"
    
    // Create directory structure
    newDir := &Directory{
        InodeID: dirInode.ID,
        Entries: make(map[string]*DirectoryEntry),
        Parent:  fsm.currentDir,
        Name:    name,
    }
    
    // Add to parent directory
    entry := &DirectoryEntry{
        Name:    name,
        InodeID: dirInode.ID,
        Type:    "directory",
    }
    
    fsm.currentDir.Entries[name] = entry
    fsm.dirs[dirInode.ID] = newDir
    
    // Add "." and ".." entries
    newDir.Entries["."] = &DirectoryEntry{
        Name:    ".",
        InodeID: dirInode.ID,
        Type:    "directory",
    }
    
    newDir.Entries[".."] = &DirectoryEntry{
        Name:    "..",
        InodeID: fsm.currentDir.InodeID,
        Type:    "directory",
    }
    
    fmt.Printf("Created directory '%s'\n", name)
    return true
}

func (fsm *FileSystemManager) CreateFile(name string, size int64) bool {
    if _, exists := fsm.currentDir.Entries[name]; exists {
        fmt.Printf("File '%s' already exists\n", name)
        return false
    }
    
    // Create inode for file
    fileInode := fsm.fs.CreateInode(0644, 1000, 1000)
    fileInode.Type = "file"
    fsm.fs.UpdateInode(fileInode.ID, size)
    
    // Add to current directory
    entry := &DirectoryEntry{
        Name:    name,
        InodeID: fileInode.ID,
        Type:    "file",
    }
    
    fsm.currentDir.Entries[name] = entry
    
    fmt.Printf("Created file '%s' (size: %d bytes)\n", name, size)
    return true
}

func (fsm *FileSystemManager) ChangeDirectory(path string) bool {
    if path == "/" {
        fsm.currentDir = fsm.rootDir
        fmt.Println("Changed to root directory")
        return true
    }
    
    if path == ".." {
        if fsm.currentDir.Parent != nil {
            fsm.currentDir = fsm.currentDir.Parent
            fmt.Printf("Changed to parent directory: %s\n", fsm.currentDir.Name)
            return true
        }
        fmt.Println("Already at root directory")
        return false
    }
    
    if entry, exists := fsm.currentDir.Entries[path]; exists {
        if entry.Type == "directory" {
            if dir, exists := fsm.dirs[entry.InodeID]; exists {
                fsm.currentDir = dir
                fmt.Printf("Changed to directory: %s\n", path)
                return true
            }
        } else {
            fmt.Printf("'%s' is not a directory\n", path)
        }
    } else {
        fmt.Printf("Directory '%s' not found\n", path)
    }
    
    return false
}

func (fsm *FileSystemManager) ListDirectory() {
    fmt.Printf("Contents of directory '%s':\n", fsm.currentDir.Name)
    fmt.Println("Type\tName\t\tInode ID")
    fmt.Println("----\t----\t\t--------")
    
    for name, entry := range fsm.currentDir.Entries {
        if name != "." && name != ".." {
            fmt.Printf("%s\t%s\t\t%d\n", entry.Type, name, entry.InodeID)
        }
    }
}

func (fsm *FileSystemManager) RemoveFile(name string) bool {
    if entry, exists := fsm.currentDir.Entries[name]; exists {
        if entry.Type == "file" {
            delete(fsm.currentDir.Entries, name)
            fmt.Printf("Removed file '%s'\n", name)
            return true
        } else {
            fmt.Printf("'%s' is a directory, use RemoveDirectory\n", name)
        }
    } else {
        fmt.Printf("File '%s' not found\n", name)
    }
    
    return false
}

func (fsm *FileSystemManager) RemoveDirectory(name string) bool {
    if entry, exists := fsm.currentDir.Entries[name]; exists {
        if entry.Type == "directory" {
            if dir, exists := fsm.dirs[entry.InodeID]; exists {
                // Check if directory is empty (only . and ..)
                if len(dir.Entries) <= 2 {
                    delete(fsm.currentDir.Entries, name)
                    delete(fsm.dirs, entry.InodeID)
                    fmt.Printf("Removed directory '%s'\n", name)
                    return true
                } else {
                    fmt.Printf("Directory '%s' is not empty\n", name)
                }
            }
        } else {
            fmt.Printf("'%s' is a file, use RemoveFile\n", name)
        }
    } else {
        fmt.Printf("Directory '%s' not found\n", name)
    }
    
    return false
}

func (fsm *FileSystemManager) GetCurrentPath() string {
    if fsm.currentDir == fsm.rootDir {
        return "/"
    }
    
    path := ""
    current := fsm.currentDir
    
    for current != nil && current != fsm.rootDir {
        path = "/" + current.Name + path
        current = current.Parent
    }
    
    return "/" + path
}

func main() {
    fs := NewFileSystem(4096)
    fsm := NewFileSystemManager(fs)
    
    fmt.Println("File System Manager Demo:")
    
    // Create some directories and files
    fsm.CreateDirectory("home")
    fsm.CreateDirectory("etc")
    fsm.CreateFile("README.txt", 1024)
    
    fsm.ListDirectory()
    
    // Change to home directory
    fsm.ChangeDirectory("home")
    fsm.CreateDirectory("user1")
    fsm.CreateDirectory("user2")
    fsm.CreateFile("config.ini", 512)
    
    fsm.ListDirectory()
    
    // Change to user1 directory
    fsm.ChangeDirectory("user1")
    fsm.CreateFile("document.txt", 2048)
    fsm.CreateFile("image.jpg", 4096)
    
    fsm.ListDirectory()
    
    // Go back to parent
    fsm.ChangeDirectory("..")
    fsm.ListDirectory()
    
    // Show current path
    fmt.Printf("Current path: %s\n", fsm.GetCurrentPath())
}
```

## File Operations

### Theory

File operations include reading, writing, seeking, and truncating files. These operations work with the file system's inode structure and data blocks.

### File Operations Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "io"
    "time"
)

type File struct {
    InodeID   int
    Position  int64
    Mode      string
    FileSystem *FileSystem
}

type FileOperations struct {
    fs *FileSystem
    openFiles map[int]*File
    nextFD    int
}

func NewFileOperations(fs *FileSystem) *FileOperations {
    return &FileOperations{
        fs:        fs,
        openFiles: make(map[int]*File),
        nextFD:    1,
    }
}

func (fo *FileOperations) Open(inodeID int, mode string) int {
    if _, exists := fo.fs.inodes[inodeID]; !exists {
        fmt.Printf("Inode %d not found\n", inodeID)
        return -1
    }
    
    file := &File{
        InodeID:    inodeID,
        Position:   0,
        Mode:       mode,
        FileSystem: fo.fs,
    }
    
    fd := fo.nextFD
    fo.openFiles[fd] = file
    fo.nextFD++
    
    fmt.Printf("Opened file (inode %d) with fd %d in mode '%s'\n", inodeID, fd, mode)
    return fd
}

func (fo *FileOperations) Close(fd int) bool {
    if file, exists := fo.openFiles[fd]; exists {
        delete(fo.openFiles, fd)
        fmt.Printf("Closed file with fd %d\n", fd)
        return true
    }
    
    fmt.Printf("File descriptor %d not found\n", fd)
    return false
}

func (fo *FileOperations) Read(fd int, buffer []byte) (int, error) {
    file, exists := fo.openFiles[fd]
    if !exists {
        return 0, fmt.Errorf("file descriptor %d not found", fd)
    }
    
    if file.Mode == "w" {
        return 0, fmt.Errorf("file opened in write-only mode")
    }
    
    inode, exists := fo.fs.inodes[file.InodeID]
    if !exists {
        return 0, fmt.Errorf("inode %d not found", file.InodeID)
    }
    
    if file.Position >= inode.Size {
        return 0, io.EOF
    }
    
    bytesRead := 0
    remaining := int(inode.Size - file.Position)
    toRead := len(buffer)
    if toRead > remaining {
        toRead = remaining
    }
    
    // Read from direct blocks
    blockSize := int64(fo.fs.blockSize)
    currentPos := file.Position
    
    for bytesRead < toRead {
        blockIndex := currentPos / blockSize
        blockOffset := currentPos % blockSize
        
        if blockIndex >= 12 {
            // Would need to handle indirect blocks in a real implementation
            break
        }
        
        blockID := inode.DirectBlocks[blockIndex]
        if blockID == 0 {
            break
        }
        
        // Calculate how much to read from this block
        blockRemaining := blockSize - blockOffset
        needToRead := toRead - bytesRead
        readFromBlock := int(blockRemaining)
        if readFromBlock > needToRead {
            readFromBlock = needToRead
        }
        
        // Read data from block
        data := fo.fs.ReadFromBlock(blockID, int(blockOffset), readFromBlock)
        copy(buffer[bytesRead:], data)
        
        bytesRead += len(data)
        currentPos += int64(len(data))
        
        if len(data) < readFromBlock {
            break
        }
    }
    
    file.Position = currentPos
    inode.Atime = time.Now()
    
    fmt.Printf("Read %d bytes from fd %d\n", bytesRead, fd)
    return bytesRead, nil
}

func (fo *FileOperations) Write(fd int, data []byte) (int, error) {
    file, exists := fo.openFiles[fd]
    if !exists {
        return 0, fmt.Errorf("file descriptor %d not found", fd)
    }
    
    if file.Mode == "r" {
        return 0, fmt.Errorf("file opened in read-only mode")
    }
    
    inode, exists := fo.fs.inodes[file.InodeID]
    if !exists {
        return 0, fmt.Errorf("inode %d not found", file.InodeID)
    }
    
    bytesWritten := 0
    blockSize := int64(fo.fs.blockSize)
    currentPos := file.Position
    
    for bytesWritten < len(data) {
        blockIndex := currentPos / blockSize
        blockOffset := currentPos % blockSize
        
        if blockIndex >= 12 {
            // Would need to handle indirect blocks in a real implementation
            break
        }
        
        // Allocate block if needed
        if inode.DirectBlocks[blockIndex] == 0 {
            inode.DirectBlocks[blockIndex] = fo.fs.AllocateBlock()
        }
        
        blockID := inode.DirectBlocks[blockIndex]
        
        // Calculate how much to write to this block
        blockRemaining := blockSize - blockOffset
        needToWrite := len(data) - bytesWritten
        writeToBlock := int(blockRemaining)
        if writeToBlock > needToWrite {
            writeToBlock = needToWrite
        }
        
        // Write data to block
        fo.fs.WriteToBlock(blockID, data[bytesWritten:bytesWritten+writeToBlock], int(blockOffset))
        
        bytesWritten += writeToBlock
        currentPos += int64(writeToBlock)
    }
    
    file.Position = currentPos
    
    // Update inode size if we wrote beyond current size
    if currentPos > inode.Size {
        inode.Size = currentPos
        inode.Blocks = (currentPos + blockSize - 1) / blockSize
    }
    
    inode.Mtime = time.Now()
    inode.Ctime = time.Now()
    
    fmt.Printf("Wrote %d bytes to fd %d\n", bytesWritten, fd)
    return bytesWritten, nil
}

func (fo *FileOperations) Seek(fd int, offset int64, whence int) (int64, error) {
    file, exists := fo.openFiles[fd]
    if !exists {
        return 0, fmt.Errorf("file descriptor %d not found", fd)
    }
    
    inode, exists := fo.fs.inodes[file.InodeID]
    if !exists {
        return 0, fmt.Errorf("inode %d not found", file.InodeID)
    }
    
    var newPos int64
    
    switch whence {
    case 0: // SEEK_SET
        newPos = offset
    case 1: // SEEK_CUR
        newPos = file.Position + offset
    case 2: // SEEK_END
        newPos = inode.Size + offset
    default:
        return 0, fmt.Errorf("invalid whence value")
    }
    
    if newPos < 0 {
        newPos = 0
    }
    if newPos > inode.Size {
        newPos = inode.Size
    }
    
    file.Position = newPos
    fmt.Printf("Seeked to position %d in fd %d\n", newPos, fd)
    return newPos, nil
}

func (fo *FileOperations) Truncate(fd int, size int64) error {
    file, exists := fo.openFiles[fd]
    if !exists {
        return fmt.Errorf("file descriptor %d not found", fd)
    }
    
    inode, exists := fo.fs.inodes[file.InodeID]
    if !exists {
        return fmt.Errorf("inode %d not found", file.InodeID)
    }
    
    if size < 0 {
        return fmt.Errorf("invalid size")
    }
    
    inode.Size = size
    inode.Mtime = time.Now()
    inode.Ctime = time.Now()
    
    // Update block count
    blockSize := int64(fo.fs.blockSize)
    inode.Blocks = (size + blockSize - 1) / blockSize
    
    // Adjust position if it's beyond new size
    if file.Position > size {
        file.Position = size
    }
    
    fmt.Printf("Truncated file (fd %d) to size %d\n", fd, size)
    return nil
}

func main() {
    fs := NewFileSystem(4096)
    fo := NewFileOperations(fs)
    
    fmt.Println("File Operations Demo:")
    
    // Create a file
    inode := fs.CreateInode(0644, 1000, 1000)
    fs.UpdateInode(inode.ID, 0)
    
    // Open file for writing
    fd := fo.Open(inode.ID, "w")
    
    // Write some data
    data := []byte("Hello, World! This is a test file.")
    fo.Write(fd, data)
    
    // Seek to beginning
    fo.Seek(fd, 0, 0)
    
    // Read data back
    buffer := make([]byte, len(data))
    bytesRead, err := fo.Read(fd, buffer)
    if err != nil {
        fmt.Printf("Error reading: %v\n", err)
    } else {
        fmt.Printf("Read: %s\n", string(buffer[:bytesRead]))
    }
    
    // Truncate file
    fo.Truncate(fd, 10)
    
    // Read truncated data
    buffer = make([]byte, 20)
    bytesRead, err = fo.Read(fd, buffer)
    if err != nil {
        fmt.Printf("Error reading: %v\n", err)
    } else {
        fmt.Printf("Read truncated: %s\n", string(buffer[:bytesRead]))
    }
    
    // Close file
    fo.Close(fd)
}
```

## Follow-up Questions

### 1. Inode Structure
**Q: What are the advantages of using inodes over simple file allocation tables?**
A: Inodes provide better performance for large files through indirect addressing, better security through permission bits, and more efficient storage of metadata. They also enable hard links and better support for sparse files.

### 2. Directory Management
**Q: How do you handle directory traversal efficiently in a file system?**
A: Use path caching to store frequently accessed directory paths, implement directory entry caching, and use efficient data structures like B-trees for large directories. Also, minimize disk I/O by batching operations.

### 3. File Operations
**Q: What are the trade-offs between synchronous and asynchronous file operations?**
A: Synchronous operations are simpler to implement and debug but can block the calling process. Asynchronous operations provide better performance and responsiveness but require more complex error handling and state management.

## Complexity Analysis

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Inode Lookup | O(1) | O(1) | Hash table lookup |
| Directory Traversal | O(n) | O(1) | Linear search through entries |
| File Read/Write | O(blocks) | O(1) | Depends on number of blocks |
| Directory Creation | O(1) | O(1) | Constant time operations |

## Applications

1. **Inode Structure**: Unix-like file systems (ext4, XFS, ZFS)
2. **Directory Management**: File system implementations, database systems
3. **File Operations**: Operating systems, database engines, web servers

---

**Next**: [I/O Systems](./io-systems.md) | **Previous**: [OS Deep Dive](../README.md) | **Up**: [OS Deep Dive](../README.md)
