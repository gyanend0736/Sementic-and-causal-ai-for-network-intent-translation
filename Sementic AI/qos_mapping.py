KEYWORD_MAP = {
    
    "game":             "gaming",
    "gaming":           "gaming",
    "play":             "gaming",
    "playing":          "gaming",
    "lag":              "gaming",
    "steam":            "gaming",
    "valorant":         "gaming",
    
    "zoom":             "zoom",
    "meeting":          "zoom",
    "call":             "zoom",
    "class":            "zoom",
    
    "netflix":          "netflix",
    "movie":            "netflix",
    "watch":            "netflix",
    "stream":           "netflix",
    "youtube":          "netflix",
    
    "web":              "browsing",
    "internet":         "browsing",
    "browse":           "browsing",
    "chrome":           "browsing"
}
APP_REQUIREMENTS = {
    "gaming": {
        "priority": 1,
        "max_latency": 30,     
        "max_jitter": 5,        
        "min_bandwidth": 5,     
        "protocol": "UDP"
    },
    "zoom": {
        "priority": 2,
        "max_latency": 100,
        "max_jitter": 30,
        "min_bandwidth": 2,
        "protocol": "UDP"
    },
    "netflix": {
        "priority": 3,
        "max_latency": 500,    
        "max_jitter": 50,
        "min_bandwidth": 25,   
        "protocol": "TCP"
    },
    "browsing": {
        "priority": 4,
        "max_latency": 1000,
        "max_jitter": 100,
        "min_bandwidth": 1,
        "protocol": "TCP"
    }
}


