module lcd_monitor(

    input logic clk,
    input logic rstb, 
    input logic [2:0] what_to_print,

    output logic [7:0] print_pattern 
);

logic [7:0] word_rom [0:7][0:7]; 
logic [2:0] char_index;
initial begin
    word_rom[0] = {"W", "E", "L", "C", "O", "M", "E", " "}; // WELCOME
    word_rom[1] = {"R", "E", "C", "O", "R", "D", " ", " "}; // RECORD
    word_rom[2] = {"U", "P", " ", " ", " ", " ", " ", " "}; // UP
    word_rom[3] = {"D", "O", "W", "N", " ", " ", " ", " "};
    word_rom[4] = {"L", "E", "F", "T", " ", " ", " ", " "};
    word_rom[5] = {"R", "I", "G", "H", "T", " ", " ", " "};
    word_rom[6] = {"S", "T", "O", "P", " ", " ", " ", " "};
    word_rom[7] = {"-", "-", "-", "-", " ", " ", " ", " "}; // SILENCE
end

logic [23:0] delay_counter;

always_ff @(posedge clk or negedge rstb) begin
    if(~rstb) begin
        char_index    <= 3'd0;
        print_pattern <= 8'd0;
        delay_counter <= 24'd0;
    end
    else begin
    if (delay_counter >= 24'd5_000_000) begin        
        print_pattern <= word_rom[what_to_print][char_index];
        delay_counter <= 24'd0;
        if (char_index == 3'd7) 
          char_index <= 3'd0;  
        else 
            char_index <=  char_index + 3'd1;
             end
    else  
        delay_counter <= delay_counter +  24'd01;
    

    end
    end
    






endmodule //lcd_monitor




module lcd_driver (
    input  logic        clk,
    input  logic        rstb,
    input  logic [7:0]  char_in,
    input  logic        char_valid,
    
    output logic [7:0]  lcd_data,
    output logic        lcd_rs,
    output logic        lcd_rw,
    output logic        lcd_en,
    
    output logic        char_ready  // אומר שאתה מוכן לקבל תו חדש
);


endmodule //lcd_driver

