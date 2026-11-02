VOICE_SETTINGS = {
    "stt": {
        "model": "nova-3",
        "language": "multi",
        "extra_kwargs": {
            "smart_format": True,
            "numerals": True,
        },
    },
    "tts": {
        "model": "sonic-3",
        "voice": "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        "language": "en",
    },
    "endpointing": {
        "mode": "fixed",
        "min_delay": 0.7,
        "max_delay": 2.5,
    },
    "identity_endpointing": {
        "min_delay": 1.2,
        "max_delay": 3.2,
    },
    "interruption": {
        "mode": "adaptive",
        "min_duration": 0.5,
        "min_words": 1,
        "false_interruption_timeout": 2.0,
        "resume_false_interruption": True,
    },
    "preemptive_generation": {
        "enabled": True,
        "preemptive_tts": False,
        "max_speech_duration": 8.0,
        "max_retries": 3,
    },
    "noise_cancellation": {
        "enabled": False,
        "provider": "ai_coustics",
        "model": "QUAIL_L",
    },
}
