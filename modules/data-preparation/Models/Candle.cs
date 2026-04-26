using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace LSTM_DATA_SHAPER.Models
{
    public class Candle
    {
        [JsonPropertyName("timestamp")]
        public long Timestamp { get; set; }

        [JsonPropertyName("gmtoffset")]
        public int? GmtOffset { get; set; }

        [JsonPropertyName("datetime")]
        public string? DateTime { get; set; }

        [JsonPropertyName("open")]
        public double? Open { get; set; }

        [JsonPropertyName("high")]
        public double? High { get; set; }

        [JsonPropertyName("low")]
        public double? Low { get; set; }

        [JsonPropertyName("close")]
        public double? Close { get; set; }

        [JsonPropertyName("volume")]
        public long? Volume { get; set; }
        public bool? MissingFlag { get; set; } = false;

        //CANDLESS
        public double? BODYTOCLOSE { get; set; }
        public double? LOWTOCLOSE { get; set; }
        public double? HIGHTOCLOSE { get; set; }

        //WYNIKOWE
        public double? ATRTOCLOSE { get; set; }
        public double? CloseVsSma10 { get; set; }
        public double? CloseVsSma20 { get; set; }
        public double? CloseVsSma50 { get; set; }
        public double? Sma10VsSma20 { get; set; }
        public double? Sma20VsSma50 { get; set; }
        public double? VROCLOG { get; set; }
        public double? RSI { get; set; }
        public double? ADXRATIO { get; set; }
        public double? PVOH { get; set; }
        public double? MASSI { get; set; }
        public double? LOGRETURN { get; set; }
        public double? CLOSETOVWAP { get; set; }
        public double? CLOSETOAVWAP { get; set; }
        public double? AILLIQLOG { get; set; }
        public double? SKEW { get; set; }
        public double? KER { get; set; }

        //predsy
        public int? EventDirection5m { get; set; }
        public int? EventDirection15m { get; set; }
    }
}