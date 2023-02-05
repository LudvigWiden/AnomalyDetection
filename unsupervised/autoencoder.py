import torch

class CAE4(torch.nn.Module):
    """
    A convolutional autoencoder with a four convolutional layers in the encoder, 
    a code layer and decoder to reconstruct the data.
    ...

    Attributes
    ----------
    C_in : int
        Number of channels in the input image
    amp : int
        Amplifies the number of channels produced by the first convolution
    latent_dim : int
        Dimension of the latent space (code layer)

    Methods
    -------
    Forward:
        Defines how the model is going to be run, from input to output
    """
    def __init__(self, C_in, amp, latent_dim):
        super().__init__()        
        C0_in = C_in
        C1_in = C0_in*amp
        C2_in = C1_in*2
        C3_in = C2_in*2
        C4_in = C3_in*2
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(C0_in, C1_in, 3, stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(C1_in, C2_in, 3, stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(C2_in, C3_in, 3, stride=2, padding=0),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(C3_in, C4_in, 3, stride=2, padding=0),
            torch.nn.ReLU(True),
        )
        
        self.latent_space = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(7 * 7 * C4_in, latent_dim)
        )
        
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 7 * 7 * C4_in),
            torch.nn.Unflatten(dim=1, unflattened_size=(C4_in, 7, 7)),
            torch.nn.ConvTranspose2d(C4_in, C3_in, 3, stride=2, output_padding=0),
            torch.nn.BatchNorm2d(C3_in),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(C3_in, C2_in, 3, stride=2, output_padding=0),
            torch.nn.BatchNorm2d(C2_in),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(C2_in, C1_in, 3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(C1_in),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(C1_in, C0_in, 3, stride=2, padding=1, output_padding=1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        #print("0 ", x.shape)
        encoded = self.encoder(x)
        #print("1 ", encoded.shape)
        z = self.latent_space(encoded)
        #print("2", z.shape)
        decoded = self.decoder(z)
        #print("2 ", decoded.shape)
        return decoded
    
    

class CAE3(torch.nn.Module):
    """
    A convolutional autoencoder with a three convolutional layers in the encoder, 
    a code layer and decoder to reconstruct the data.
    ...

    Attributes
    ----------
    C_in : int
        Number of channels in the input image
    amp : int
        Amplifies the number of channels produced by the first convolution
    latent_dim : int
        Dimension of the latent space (code layer)

    Methods
    -------
    Forward:
        Defines how the model is going to be run, from input to output
    """
    def __init__(self, C_in, amp, latent_dim):
        super().__init__()
        C0_in = C_in
        C1_in = C0_in*amp
        C2_in = C1_in*2
        C3_in = C2_in*2
               
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(C0_in, C1_in, 3, stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(C1_in, C2_in, 3, stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(C2_in, C3_in, 3, stride=2, padding=0),
            torch.nn.ReLU(True),
        )
        
        self.latent_space = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(15 * 15 * C3_in, latent_dim)
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 15 * 15 * C3_in),
            torch.nn.Unflatten(dim=1, unflattened_size=(C3_in, 15, 15)),
            torch.nn.ConvTranspose2d(C3_in, C2_in, 3, stride=2, output_padding=0),
            torch.nn.BatchNorm2d(C2_in),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(C2_in, C1_in, 3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(C1_in),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(C1_in, C0_in, 3, stride=2, padding=1, output_padding=1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        #print("0 ", x.shape)
        encoded = self.encoder(x)
        #print("1 ", encoded.shape)
        z = self.latent_space(encoded)
        #print("2", z.shape)
        decoded = self.decoder(z)
        #print("3 ", decoded.shape)
        return decoded
        

class CAEM(torch.nn.Module):
    """
    A convolutional autoencoder with a three convolutional layers and 
    three max pooling layers in the encoder, a code layer and decoder to reconstruct the data.
    ...

    Attributes
    ----------
    C_in : int
        Number of channels in the input image
    amp : int
        Amplifies the number of channels produced by the first convolution
    latent_dim : int
        Dimension of the latent space (code layer)

    Methods
    -------
    Forward:
        Defines how the model is going to be run, from input to output
    """
    def __init__(self, C_in, amp, latent_dim):
        super().__init__()
        C0_in = C_in
        C1_in = C0_in*amp
        C2_in = C1_in*2
        C3_in = C2_in*2
               
        self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=C0_in, out_channels=C1_in, kernel_size=3, stride=1, padding = 1),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(in_channels=C1_in, out_channels=C2_in, kernel_size=3, stride=1, padding = 1), 
                torch.nn.MaxPool2d(kernel_size=2, stride=2), 
                torch.nn.ReLU(True),
                torch.nn.Conv2d(in_channels=C2_in, out_channels=C3_in, kernel_size=3, stride=1, padding = 1), 
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.ReLU(True),           
            )
        
        self.latent_space = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(15*15*C3_in, latent_dim)
        )
     
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 15*15*C3_in),
            torch.nn.Unflatten(dim=1, unflattened_size=(C3_in, 15, 15)),
            torch.nn.ConvTranspose2d(in_channels=C3_in, out_channels=C2_in, kernel_size=3, stride=2), 
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(C2_in),
            torch.nn.ConvTranspose2d(in_channels=C2_in, out_channels=C1_in, kernel_size=3, stride=2, padding=1), 
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(C1_in),
            torch.nn.ConvTranspose2d(in_channels=C1_in, out_channels=C0_in, kernel_size=3, stride=2, output_padding=1), 
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(C0_in),
            torch.nn.Sigmoid()
        )
        
        
    def forward(self, x):
        #print("0 ", x.shape)
        encoded = self.encoder(x)
        #print("1 ", encoded.shape)
        z = self.latent_space(encoded)
        #print("2 ", z.shape.shape)
        decoded = self.decoder(z)
        #print("3 ", decoded.shape)
        return decoded      
        

class CAE0(torch.nn.Module):
    """
    A convolutional autoencoder with a three convolutional layers in the encoder,
    and decoder to reconstruct the data.
    ...

    Attributes
    ----------
    C_in : int
        Number of channels in the input image
    amp : int
        Amplifies the number of channels produced by the first convolution
        
    Methods
    -------
    Forward:
        Defines how the model is going to be run, from input to output
    """
    def __init__(self, C_in, amp, latent_dim):
        super().__init__()        
        C0_in = C_in
        C1_in = C0_in*amp
        C2_in = C1_in*2
        C3_in = C2_in*2
                
        self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=C0_in, out_channels=C1_in, kernel_size=3, stride=2, padding = 1), 
                torch.nn.ReLU(True),
                torch.nn.Conv2d(in_channels=C1_in, out_channels=C2_in, kernel_size=3, stride=2, padding = 1), 
                torch.nn.ReLU(True),
                torch.nn.Conv2d(in_channels=C2_in, out_channels=C3_in, kernel_size=3, stride=2, padding = 1), 
                torch.nn.ReLU(True)
                )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=C3_in, out_channels=C2_in, kernel_size=3, stride=2, padding=1), 
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(C2_in),     
            torch.nn.ConvTranspose2d(in_channels=C2_in, out_channels=C1_in, kernel_size=3, stride=2, padding=1), 
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(C1_in),
            torch.nn.ConvTranspose2d(in_channels=C1_in, out_channels=C0_in, kernel_size=3, stride=2, output_padding=1), 
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(C0_in), 
            torch.nn.Sigmoid()
            )
               
        
    def forward(self, x):
        #print("0 ", x.shape)
        encoded = self.encoder(x)
        #print("1 ", encoded.shape)
        decoded = self.decoder(encoded)
        #print("2 ", decoded.shape)
        return decoded       

