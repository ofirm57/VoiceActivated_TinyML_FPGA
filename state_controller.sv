 module state_controller(
    input logic clk,
    input logic rstb, // 
    input logic [2:0] cmd,
    input logic push_butten_in, //KEY0

    output logic  out_rec_2_arm,
    output logic [7:0] out
);


assign out_rec_2_arm = (cs == RECORDING);


typedef enum logic [2:0]{
    WELCOME =    3'd0  ,
    RECORDING =  3'd1,
    UP =         3'd2    ,
    DOWN =       3'd3   ,
    LEFT =       3'd4   ,
    RIGHT =      3'd5,
    STOP=        3'd6,
    SILENCE =    3'd7
    }state;

state cs,ns;





/////////////////////////////////////////////////// one push butten -> rec
logic push_butten_delay, rec;

always_ff @(posedge clk or negedge rstb) begin
    if(~rstb)
        push_butten_delay <= 1'b1;
    else begin
        push_butten_delay <= push_butten_in;
    end 
    end

assign rec = ~(push_butten_delay) && push_butten_in ;
/////////////////////////////////////////////////

// fsm 
always_comb begin
    case (cmd)
    WELCOME   :    ns = rec ? RECORDING : WELCOME ;
    RECORDING :    ns = cmd ;
    UP        :    ns = cmd ;
    DOWN      :    ns = cmd ;
    LEFT      :    ns = cmd ; 
    RIGHT     :    ns = cmd ;    
    STOP      :    ns = cmd ;    
    SILENCE   :    ns = cmd ;        
    default   :    ns = WELCOME ; 
        
    endcase
end

always_ff @(posedge clk or negedge rstb) begin
    if(~rstb)
        cs <= 3'd0;
    else begin
        cs <= ns;
    end
    
end

endmodule


