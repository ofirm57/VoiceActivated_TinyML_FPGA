 module top(
    input logic clk,
    input logic rstb, // 
    input logic [2:0] cmd,
    input logic push_butten_in, //KEY0

    //output logic  out_rec_2_arm,

    output logic [7:0] print_pattern,


    //input logic  char_valid,
    
    output logic [7:0]  lcd_data,
    output logic  lcd_rs,
    output logic  lcd_rw,
    output logic  lcd_en,
    output logic  char_ready
);


//logic  out_rec_2_arm;
logic  [2:0] what_to_print;

state_controller the_state (.clk(clk), .rstb(rstb), .cmd(cmd), .push_butten_in(push_butten_in), 
.print_out(what_to_print))


logic  [7:0] the_char;

lcd_monitor l_monitor (.clk(clk), .rstb(rstb), .what_to_print(what_to_print), .print_pattern(the_char));
assign char_valid = 1'b1;

lcd_driver driver_lcd (.clk(clk),
                        .rstb(rstb),
                        .char_in(the_char),
                        .char_valid(char_valid),
                        .lcd_data(lcd_data),
                        .lcd_rs(lcd_rs) ,
                        .lcd_rw(lcd_rw),
                        .lcd_en(lcd_en),
                        .char_ready(char_ready)
 );



endmodule//top 
